# Voice AI Agent — Nova Sonic

A real-time voice assistant powered by [Amazon Bedrock Nova Sonic](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html). The browser captures microphone audio, streams it to a FastAPI server over WebSockets, and plays back the AI-generated speech response in real time.

## Architecture

```
Browser (Web Audio API)
  ├── /ws/user_input   →  16 kHz PCM16 base64  →  FastAPI  →  Bedrock Nova Sonic
  └── /ws/speech_output ←  24 kHz PCM16 bytes   ←  FastAPI  ←  Bedrock Nova Sonic
```

## Project Structure

```
voice-ai-nova/
├── app/
│   ├── server/
│   │   ├── main.py           # FastAPI server, WebSocket endpoints
│   │   └── nova_bridge.py    # Bedrock Nova Sonic bidirectional streaming client
│   └── static/
│       └── index.html        # Browser UI with mic capture and audio playback
├── requirements.txt
├── .env                      # Local config (not committed)
└── README.md
```

## Prerequisites

- Python 3.10+
- An AWS account with [Bedrock Nova Sonic](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html) model access enabled in your target region
- A browser that supports the Web Audio API (Chrome, Edge, Firefox)

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd voice-ai-nova
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\Activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy the example below into a `.env` file at the project root:

   ```env
   # Server
   HOST=0.0.0.0
   PORT=8000

   # AWS credentials
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key

   # Nova Sonic
   NOVA_MODEL_ID=amazon.nova-sonic-v1:0
   NOVA_VOICE_ID=tiffany

   # Audio (defaults usually work fine)
   AUDIO_SAMPLE_RATE=16000
   AUDIO_BITS=16
   AUDIO_CHANNELS=1
   ```

   | Variable | Default | Description |
   |---|---|---|
   | `HOST` | `0.0.0.0` | Server bind address |
   | `PORT` | `8000` | Server port |
   | `AWS_REGION` | `us-east-1` | AWS region with Bedrock access |
   | `AWS_ACCESS_KEY_ID` | — | AWS access key |
   | `AWS_SECRET_ACCESS_KEY` | — | AWS secret key |
   | `NOVA_MODEL_ID` | `amazon.nova-sonic-v1:0` | Bedrock model ID |
   | `NOVA_VOICE_ID` | `tiffany` | Voice for speech output (`matthew`, `tiffany`, etc.) |
   | `AUDIO_SAMPLE_RATE` | `16000` | Mic input sample rate (Hz) |
   | `AUDIO_BITS` | `16` | Audio bit depth |
   | `AUDIO_CHANNELS` | `1` | Audio channels (mono) |

## Running

```bash
python -m uvicorn app.server.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser, grant microphone access, and click **Start Recording** to begin a conversation.

## API Endpoints

| Endpoint | Type | Description |
|---|---|---|
| `GET /` | HTTP | Serves the voice UI |
| `GET /health` | HTTP | Health check (`{"status": "ok"}`) |
| `WS /ws/user_input` | WebSocket | Receives base64 PCM16 audio from the browser |
| `WS /ws/speech_output` | WebSocket | Streams PCM16 audio bytes back to the browser |
