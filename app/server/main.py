import os
import json
import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from websockets.exceptions import ConnectionClosedOK

# Nova Sonic bridge
from .nova_bridge import NovaSession

# App setup
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Nova Sonic Voice Server", version="1.0.0")

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Audio parameters must match frontend & Nova input config
SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))  # mic → Nova input (16k)
BITS = int(os.getenv("AUDIO_BITS", "16"))
CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))

# Nova Sonic typical output is 24 kHz PCM
OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", "24000"))

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful, fast voice assistant. Keep your responses short and conversational, typically 1-2 sentences for chatty scenarios.")

# Shared state (single-client demo)
speech_ws: Optional[WebSocket] = None        # for binary audio bytes to the browser
nova_session: Optional[NovaSession] = None   # active Nova Sonic session
read_task: Optional[asyncio.Task] = None     # background task reading Nova stream
input_started: bool = False                  # whether we've opened AUDIO content block

CHUNK_SIZE = 1024  # Audio chunk size for streaming

# Routes
@app.get("/")
async def index():
    """Serve your separate index.html file."""
    import os
    from pathlib import Path

    current_dir = Path(__file__).parent  # app/server
    static_dir = current_dir.parent / "static"  # app/static
    index_file = static_dir / "index.html"

    try:
        # For organized structure - adjust path as needed
        with open(str(index_file), "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        # Fallback to root directory
        try:
            with open("index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
        except FileNotFoundError:
            return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/health")
async def health():
    return {"status": "ok"}

# Utility functions
async def _send_speech_bytes(b: bytes) -> None:
    """Send raw PCM audio bytes to the browser over /ws/speech_output."""
    global speech_ws
    if speech_ws is None:
        return
    try:
        await speech_ws.send_bytes(b)
    except Exception as e:
        logging.warning("Failed to send audio bytes to client: %s", e)

async def _send_control(msg: dict) -> None:
    """Send a small JSON control message to the browser."""
    global speech_ws
    if speech_ws is None:
        return
    try:
        await speech_ws.send_text(json.dumps(msg))
    except Exception as e:
        logging.warning("Failed to send control message to client: %s", e)

@app.websocket("/ws/speech_output")
async def speech_output_ws(ws: WebSocket):
    """
    Socket used to stream agent audio (PCM bytes) to the browser.
    """
    global speech_ws
    await ws.accept()
    speech_ws = ws
    logging.info("Speech output WebSocket connected")
    
    # Let the client know the output sample rate
    try:
        await _send_control({"status": "connected", "output_sample_rate": OUTPUT_SAMPLE_RATE})
    except Exception:
        pass

    try:
        while True:
            # Keep alive; no inbound messages expected on this socket
            await asyncio.sleep(30)
            if ws.client_state.name != "CONNECTED":
                break
    except WebSocketDisconnect:
        logging.info("Speech output WebSocket disconnected")
    except Exception as e:
        logging.error(f"Speech output WebSocket error: {e}")
    finally:
        speech_ws = None

@app.websocket("/ws/user_input")
async def user_input_ws(ws: WebSocket):
    """
    Socket receiving base64-encoded LPCM16 (16kHz mono) audio frames from the browser.
    """
    global nova_session, read_task, input_started

    await ws.accept()
    logging.info("User input WebSocket connected")

    async def on_audio_bytes(b: bytes):
        """Forward Nova audio to the client."""
        logging.debug(f"Received {len(b)} audio bytes from Nova")
        await _send_speech_bytes(b)

    async def on_control(msg: dict):
        """Handle control messages like barge-in"""
        logging.info(f"Received control message: {msg}")
        await _send_control(msg)

    try:
        while True:
            try:
                data = await ws.receive_text()
            except WebSocketDisconnect:
                break

            # Handle STOP command
            if data == "STOP":
                logging.info("Received STOP from client")
                if nova_session and input_started:
                    try:
                        await nova_session.end_audio_input()
                        input_started = False
                    except Exception as e:
                        logging.exception("end_audio_input failed on STOP: %s", e)
                
                if nova_session:
                    try:
                        await nova_session.end()
                    except Exception as e:
                        logging.exception("session end failed on STOP: %s", e)
                    nova_session = None
                
                await _send_control({"status": "idle"})
                continue

            # Lazy initialization of Nova session
            if nova_session is None:
                try:
                    logging.info("Creating new Nova Sonic session")
                    nova_session = NovaSession.from_env(
                        sample_rate=SAMPLE_RATE,
                        sample_size_bits=BITS,
                        channel_count=CHANNELS,
                    )
                    logging.info("Starting Nova Sonic session with system prompt")
                    await nova_session.start(system_prompt=SYSTEM_PROMPT)

                    # Start Nova -> browser reader in the background
                    logging.info("Starting background read task")
                    read_task = asyncio.create_task(
                        nova_session.read_loop(on_audio_bytes, on_control)
                    )
                    logging.info("Nova Sonic session fully initialized")
                    await _send_control({"status": "connected", "output_sample_rate": OUTPUT_SAMPLE_RATE})
                except Exception as e:
                    logging.exception("Failed to start Nova session: %s", e)
                    await _send_control({"status": "error", "detail": "nova_start_failed"})
                    continue

            # Barge-in is handled by Nova itself via interrupt events;
            # don't clear audio on every mic frame.

            # Start audio input on first frame
            if not input_started:
                try:
                    await nova_session.start_audio_input()
                    input_started = True
                    logging.info("Audio input started")
                except Exception as e:
                    logging.exception("start_audio_input failed: %s", e)
                    await _send_control({"status": "error", "detail": "audio_start_failed"})
                    continue

            # Forward base64 PCM frame to Nova
            try:
                logging.debug(f"Sending {len(data)} chars of base64 audio to Nova")
                await nova_session.send_audio_base64(data)
            except Exception as e:
                logging.exception("Failed to send audio to Nova: %s", e)

    except ConnectionClosedOK:
        logging.info("User input WebSocket closed (OK)")
    except Exception as e:
        logging.exception("user_input_ws error: %s", e)
    finally:
        # Graceful shutdown
        try:
            if nova_session:
                if input_started:
                    try:
                        await nova_session.end_audio_input()
                    except Exception:
                        pass
                try:
                    await nova_session.end()
                except Exception:
                    pass
        finally:
            nova_session = None
            input_started = False

        if read_task and not read_task.done():
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass
            read_task = None

        logging.info("User input WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")