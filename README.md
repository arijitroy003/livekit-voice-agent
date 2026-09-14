# LiveKit Voice Agent

Prototype voice assistant from the [Building Production-Ready Voice Agents with LiveKit](https://worksh.app/tutorials/livekit-voice-agent/foundations) workshop. One `agent.py` combines all six lessons:

1. Foundations — WebRTC session, Silero VAD, background voice cancellation
2. Semantic turn detection — multilingual end-of-turn model
3. Personality and fallbacks — tech-support persona, Cartesia voice, STT/LLM/TTS failover
4. Metrics and latency — usage logs, time-to-first-audio, preemptive generation
5. Tools and MCP — Open-Meteo weather plus LiveKit docs MCP
6. Consent and escalations — recording-consent task and manager handoff

Speech, language, and voice models run through [LiveKit Inference](https://docs.livekit.io/agents/integrations/inference/). You only need LiveKit Cloud credentials.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A free [LiveKit Cloud](https://cloud.livekit.io) project (API URL, key, and secret)

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env` from your LiveKit Cloud project settings:

```ini
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
```

Download local models (Silero VAD and the turn detector):

```bash
uv run agent.py download-files
```

## Run locally

Console mode uses your machine microphone and speakers:

```bash
uv run agent.py console
```

Optional device flags:

- `--list-devices` — list input and output devices
- `--input-device` — numeric input device ID
- `--output-device` — numeric output device ID

Stop with Ctrl+C. The shutdown callback prints a usage summary.

## What to try

- Pause mid-sentence for about a second. The agent should wait.
- Ask for the weather in a city, then try an invalid place such as Mars.
- Ask how to add a function tool in LiveKit (uses the docs MCP server).
- Decline recording consent, then continue the conversation.
- Ask to speak to a manager and confirm the handoff and voice change.

Cloud session recording still defaults to on for this prototype. Consent only changes what the agent says. Toggle Agent observability under LiveKit Cloud project settings, or pass `record=False` in `session.start` if you want a session skipped.

## Optional: deploy to LiveKit Cloud

```bash
brew install livekit-cli   # or see https://docs.livekit.io/home/cli
lk cloud auth
lk agent create
```

`lk agent create` writes `livekit.toml` (do not commit it) and can reuse the Dockerfile in this repo.

## Follow-ups not wired into this prototype

- **Task groups** for multi-step checkout (email, then shipping address)
- **SIP / telephony** warm handoff to a human (`publish_sip_participant` needs a real SIP trunk)

See the [consent and escalations lesson](https://worksh.app/tutorials/livekit-voice-agent/consent-and-escalations) for those patterns.

## Pipeline

```
Caller audio
  -> Silero VAD + BVC
  -> STT (AssemblyAI, then Deepgram)
  -> multilingual turn detector
  -> LLM (GPT-4.1-mini, then Gemini 2.5 Flash)
  -> weather tool / LiveKit MCP / manager handoff
  -> TTS (Cartesia Sonic 3, then Inworld)
```

Target time to first audio is under one second for most turns.
