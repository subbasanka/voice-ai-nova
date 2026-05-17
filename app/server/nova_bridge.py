# nova_bridge.py
import os
import json
import uuid
import base64
import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

# AWS Bedrock Runtime client for bidirectional streaming
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver

# Configuration from environment
MODEL_ID = os.getenv("NOVA_MODEL_ID", "amazon.nova-sonic-v1:0")
VOICE_ID = os.getenv("NOVA_VOICE_ID", "matthew")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAMPLE_RATE = int(os.getenv("NOVA_SAMPLE_RATE", "16000"))
BITS = 16
CHANNELS = 1

# Type aliases
JsonDict = dict[str, object]
OnAudioBytes = Callable[[bytes], Awaitable[None]]
OnControl = Callable[[JsonDict], Awaitable[None]]

def _event(payload: JsonDict) -> InvokeModelWithBidirectionalStreamInputChunk:
    """Wrap a JSON event for the input stream."""
    body = json.dumps({"event": payload}).encode("utf-8")
    return InvokeModelWithBidirectionalStreamInputChunk(
        value=BidirectionalInputPayloadPart(bytes_=body)
    )

@dataclass
class NovaSession:
    client: Optional[BedrockRuntimeClient] = None
    prompt_name: str = ""
    content_name: str = ""
    audio_content_name: str = ""
    active: bool = False
    stream_started: bool = False
    stream: any = None
    sample_rate: int = SAMPLE_RATE
    sample_size_bits: int = BITS
    channel_count: int = CHANNELS
    
    # Audio and event queues for background processing
    audio_queue: Optional[asyncio.Queue] = None
    event_queue: Optional[asyncio.Queue] = None
    barge_in: bool = False

    def __post_init__(self):
        """Initialize unique IDs and queues after dataclass creation."""
        if not self.prompt_name:
            self.prompt_name = str(uuid.uuid4())
        if not self.content_name:
            self.content_name = str(uuid.uuid4())
        if not self.audio_content_name:
            self.audio_content_name = str(uuid.uuid4())
        
        self.audio_queue = asyncio.Queue()
        self.event_queue = asyncio.Queue()

    @classmethod
    def from_env(cls, sample_rate: int = SAMPLE_RATE, sample_size_bits: int = BITS, channel_count: int = CHANNELS):
        """Create a NovaSession instance with Bedrock Runtime Smithy client."""
        cfg = Config(
            endpoint_uri=f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com",
            region=AWS_REGION,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        
        session = cls()
        session.client = BedrockRuntimeClient(config=cfg)
        session.sample_rate = sample_rate
        session.sample_size_bits = sample_size_bits
        session.channel_count = channel_count
        return session

    async def send_event(self, event_json: str):
        """Send an event to the stream."""
        if not self.active or not self.stream:
            return
        
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        await self.stream.input_stream.send(event)

    async def start(self, *, system_prompt: str | None = None):
        """Initialize bidirectional stream and send session/prompt bootstrap events."""
        if not self.client:
            raise RuntimeError("NovaSession client not initialized")

        try:
            # Open the bidirectional stream
            self.stream = await self.client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID)
            )
            self.active = True

            # 1) sessionStart
            session_start = {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": 1024,
                            "topP": 0.9,
                            "temperature": 0.7
                        }
                    }
                }
            }
            await self.send_event(json.dumps(session_start))

            # 2) promptStart: configure output speech + text
            prompt_start = {
                "event": {
                    "promptStart": {
                        "promptName": self.prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 24000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": VOICE_ID,
                            "encoding": "base64",
                            "audioType": "SPEECH"
                        }
                    }
                }
            }
            await self.send_event(json.dumps(prompt_start))

            # 3) Optional: system text content
            if system_prompt:
                text_content_start = {
                    "event": {
                        "contentStart": {
                            "promptName": self.prompt_name,
                            "contentName": self.content_name,
                            "type": "TEXT",
                            "interactive": True,
                            "role": "SYSTEM",
                            "textInputConfiguration": {"mediaType": "text/plain"}
                        }
                    }
                }
                await self.send_event(json.dumps(text_content_start))

                text_input = {
                    "event": {
                        "textInput": {
                            "promptName": self.prompt_name,
                            "contentName": self.content_name,
                            "content": system_prompt
                        }
                    }
                }
                await self.send_event(json.dumps(text_input))

                text_content_end = {
                    "event": {
                        "contentEnd": {
                            "promptName": self.prompt_name,
                            "contentName": self.content_name
                        }
                    }
                }
                await self.send_event(json.dumps(text_content_end))

            self.stream_started = True
            logging.info("Nova Sonic session initialized and prompt configured.")
        except Exception as e:
            logging.exception("Failed to initialize Nova Sonic session: %s", e)
            raise

    async def start_audio_input(self):
        """Send AUDIO contentStart for incoming audio chunks."""
        if not self.active or not self.stream_started:
            return

        audio_content_start = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": self.sample_rate,
                        "sampleSizeBits": self.sample_size_bits,
                        "channelCount": self.channel_count,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }
                }
            }
        }
        await self.send_event(json.dumps(audio_content_start))

    async def send_audio_base64(self, b64_pcm: str):
        """Send an audio chunk (base64) to Nova Sonic."""
        if not self.active or not self.stream_started:
            return
        try:
            audio_input = {
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": b64_pcm
                    }
                }
            }
            await self.send_event(json.dumps(audio_input))
        except Exception as e:
            logging.error("Failed to send audio chunk: %s", e)

    async def end_audio_input(self):
        """Close the current AUDIO content block."""
        if not self.active or not self.stream_started:
            return
        try:
            audio_content_end = {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name
                    }
                }
            }
            await self.send_event(json.dumps(audio_content_end))
        except Exception as e:
            logging.error("Failed to end audio input: %s", e)

    async def end(self):
        """End the session cleanly."""
        if not self.active:
            return
        try:
            # Tell the model the prompt is ending
            prompt_end = {
                "event": {
                    "promptEnd": {
                        "promptName": self.prompt_name
                    }
                }
            }
            try:
                await self.send_event(json.dumps(prompt_end))
            except Exception:
                pass

            # End the session
            session_end = {
                "event": {
                    "sessionEnd": {}
                }
            }
            try:
                await self.send_event(json.dumps(session_end))
            except Exception:
                pass

            await asyncio.sleep(0.05)
            try:
                await self.stream.input_stream.close()
            except Exception:
                pass

            self.active = False
            self.stream_started = False
            logging.info("Nova Sonic session ended.")
        finally:
            self.active = False
            self.stream_started = False

    async def _process_responses(self):
        """Process responses from Nova Sonic using the correct stream interface."""
        logging.info("Starting Nova response processor")
        try:
            while self.active:
                try:
                    # Multiple approaches to handle different SDK versions
                    result = None
                    
                    # Try the documented approach first
                    try:
                        output = await self.stream.await_output()
                        if isinstance(output, tuple) and len(output) > 1:
                            result = await output[1].receive()
                        else:
                            result = await output.receive() if hasattr(output, 'receive') else output
                    except Exception as e:
                        logging.debug(f"await_output approach failed: {e}")
                        # Try alternative approach
                        if hasattr(self.stream, 'output_stream'):
                            result = await self.stream.output_stream.receive()
                        else:
                            raise e
                    
                    # Process the result
                    if result and hasattr(result, 'value') and result.value and hasattr(result.value, 'bytes_'):
                        response_data = result.value.bytes_.decode('utf-8')
                        logging.debug(f"Received response: {response_data[:200]}...")
                        
                        try:
                            json_data = json.loads(response_data)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse JSON: {e}, data: {response_data[:100]}")
                            continue
                        
                        if 'event' in json_data:
                            # Queue the event for external processing
                            await self.event_queue.put(json.dumps(json_data))
                            
                            event = json_data['event']
                            logging.debug(f"Processing event type: {list(event.keys())}")
                            
                            # Handle text output for barge-in detection
                            if 'textOutput' in event:
                                text = event['textOutput'].get('content', '')
                                logging.info(f"Text output: {text}")
                                if '{ "interrupted" : true }' in text:
                                    logging.info("Barge-in detected, stopping audio output")
                                    self.barge_in = True
                                    continue
                            
                            # Handle audio output
                            elif 'audioOutput' in event:
                                if not self.barge_in:
                                    audio_content = event['audioOutput'].get('content', '')
                                    if audio_content:
                                        try:
                                            audio_bytes = base64.b64decode(audio_content)
                                            await self.audio_queue.put(audio_bytes)
                                            logging.debug(f"Queued {len(audio_bytes)} audio bytes")
                                        except Exception as e:
                                            logging.error(f"Error processing audio: {e}")
                            
                            # Handle other event types
                            elif 'contentStart' in event:
                                logging.info(f"Content started: {event['contentStart'].get('role', 'unknown')}")
                            elif 'contentEnd' in event:
                                logging.info("Content ended")
                            elif 'sessionStart' in event:
                                logging.info("Session started")
                            elif 'promptStart' in event:
                                logging.info("Prompt started")
                    else:
                        logging.debug(f"Received non-standard result: {type(result)}")
                
                except Exception as e:
                    err_str = str(e).lower()
                    if "closed" in err_str or "cancelled" in err_str or "invalidstate" in err_str:
                        logging.debug(f"Stream ended: {e}")
                        break
                    logging.error(f"Error processing stream event: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logging.exception(f"Nova response processing error: {e}")
        finally:
            logging.info("Nova response processor stopped")
            self.active = False
            self.stream_started = False

    async def read_loop(self, on_audio_bytes: OnAudioBytes, on_control: OnControl):
        """
        Consumer loop that forwards audio and control events to callbacks.
        This runs the background response processor and forwards events.
        """
        if not self.active or not self.stream:
            logging.error("Cannot start read_loop: session not active or stream not available")
            return

        # Start the response processor
        logging.info("Starting read_loop with background processor")
        processor_task = asyncio.create_task(self._process_responses())
        
        try:
            audio_processed = 0
            events_processed = 0
            
            while self.active:
                try:
                    # Process audio queue with timeout
                    try:
                        audio_data = await asyncio.wait_for(
                            self.audio_queue.get(),
                            timeout=0.1
                        )
                        if audio_data and not self.barge_in:
                            await on_audio_bytes(audio_data)
                            audio_processed += 1
                            if audio_processed % 10 == 0:
                                logging.debug(f"Processed {audio_processed} audio chunks")
                    except asyncio.TimeoutError:
                        pass

                    # Process event queue with timeout
                    try:
                        event_json = await asyncio.wait_for(
                            self.event_queue.get(),
                            timeout=0.1
                        )
                        if event_json:
                            events_processed += 1
                            if events_processed % 5 == 0:
                                logging.debug(f"Processed {events_processed} events")
                                
                            try:
                                event_data = json.loads(event_json)
                                if 'event' in event_data and 'textOutput' in event_data['event']:
                                    text_content = event_data['event']['textOutput'].get('content', '')
                                    if '{ "interrupted" : true }' in text_content:
                                        await on_control({"control": "clear_audio"})
                            except json.JSONDecodeError as e:
                                logging.warning(f"Failed to parse event JSON: {e}")
                    except asyncio.TimeoutError:
                        pass
                    
                    # Reset barge-in flag and clear audio queue during interruption
                    if self.barge_in:
                        cleared_count = 0
                        while not self.audio_queue.empty():
                            try:
                                self.audio_queue.get_nowait()
                                cleared_count += 1
                            except asyncio.QueueEmpty:
                                break
                        if cleared_count > 0:
                            logging.info(f"Cleared {cleared_count} audio chunks during barge-in")
                        self.barge_in = False

                except Exception as e:
                    logging.error(f"Error in read loop: {e}")
                    break

        except Exception as e:
            logging.exception(f"Nova read loop error: {e}")
        finally:
            logging.info(f"Read loop finished. Processed {audio_processed} audio chunks, {events_processed} events")
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
            
            self.active = False
            self.stream_started = False