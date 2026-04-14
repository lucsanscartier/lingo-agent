"""
agent.py — LINGO AI Phone Agent

Architecture
────────────
  Inbound call → LiveKit room
       │
       ▼
  Deepgram STT  (streaming, via livekit-agents DeepgramSTT plugin)
       │  transcript
       ▼
  HF Inference  (Qwen2.5-7B-Instruct, OpenAI-compatible chat endpoint)
       │  text reply
       ▼
  HF Inference TTS  (Kokoro-82M, returns WAV bytes)
       │  audio frames
       ▼
  LiveKit audio track → caller hears the reply

Memory
──────
  Per-caller conversation history is persisted in a local JSON file
  (see memory.py).  The phone number is extracted from the LiveKit
  participant identity or SIP metadata.

Environment Variables (set in .env or HF Spaces secrets)
─────────────────────────────────────────────────────────
  LIVEKIT_URL          wss://your-project.livekit.cloud
  LIVEKIT_API_KEY      LiveKit API key
  LIVEKIT_API_SECRET   LiveKit API secret
  DEEPGRAM_API_KEY     Deepgram API key
  HF_TOKEN             Hugging Face access token (read scope)
  MEMORY_FILE          Path to JSON memory store (default: conversation_memory.json)
  LOG_LEVEL            DEBUG | INFO | WARNING | ERROR  (default: INFO)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
from typing import AsyncIterator, Optional

import httpx
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

# Load .env before importing LiveKit so env vars are available immediately
load_dotenv()

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram

import memory
import prompts

# ── Logging ───────────────────────────────────────────────────────────────────

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("lingo.agent")

# ── Credentials (validated at start-up) ───────────────────────────────────────

REQUIRED_ENV = [
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY",
    "HF_TOKEN",
]


def _check_env() -> None:
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        logger.error(
            "Missing required environment variables: %s.  "
            "Copy .env.example to .env and fill in your credentials.",
            ", ".join(missing),
        )
        sys.exit(1)


# ── Hugging Face Inference helpers ────────────────────────────────────────────

HF_API_BASE = "https://api-inference.huggingface.co"

# LLM — Qwen2.5-7B-Instruct (free on HF Serverless Inference)
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_ENDPOINT = f"{HF_API_BASE}/models/{LLM_MODEL}/v1/chat/completions"

# TTS — Kokoro-82M (free on HF Serverless Inference)
# The /models/<id> endpoint returns raw WAV audio for TTS-capable models.
TTS_MODEL = "hexgrad/Kokoro-82M"
TTS_ENDPOINT = f"{HF_API_BASE}/models/{TTS_MODEL}"

# Audio output format that LiveKit expects for PCM frames
SAMPLE_RATE = 24_000   # 24 kHz — Kokoro default output rate
CHANNELS = 1            # mono


async def call_llm(
    messages: list[dict],
    http_client: httpx.AsyncClient,
) -> str:
    """Send a chat completion request to Qwen2.5-7B-Instruct on HF Inference.

    Args:
        messages: OpenAI-style list of message dicts.
        http_client: Shared async HTTP client.

    Returns:
        The assistant's reply text.

    Raises:
        RuntimeError: On API or network errors (caller handles gracefully).
    """
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7,
    }
    logger.debug("LLM request — %d messages", len(messages))
    try:
        resp = await http_client.post(
            LLM_ENDPOINT, json=payload, headers=headers, timeout=30.0
        )
        resp.raise_for_status()
        data = resp.json()
        text: str = data["choices"][0]["message"]["content"]
        logger.debug("LLM reply — %d chars", len(text))
        return text
    except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as exc:
        logger.error("LLM call failed: %s", exc)
        raise RuntimeError("LLM failure") from exc


async def call_tts(
    text: str,
    http_client: httpx.AsyncClient,
) -> bytes:
    """Synthesise *text* to speech using Kokoro-82M on HF Inference.

    Args:
        text: The sentence(s) to speak.
        http_client: Shared async HTTP client.

    Returns:
        Raw WAV bytes ready for decoding.

    Raises:
        RuntimeError: On API or network errors.
    """
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": text}
    logger.debug("TTS request — %d chars", len(text))
    try:
        resp = await http_client.post(
            TTS_ENDPOINT, json=payload, headers=headers, timeout=60.0
        )
        resp.raise_for_status()
        return resp.content
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.error("TTS call failed: %s", exc)
        raise RuntimeError("TTS failure") from exc


def wav_to_frames(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to a float32 numpy array.

    Args:
        wav_bytes: Raw WAV file content.

    Returns:
        ``(samples, sample_rate)`` where *samples* is a 1-D float32 array.
    """
    buf = io.BytesIO(wav_bytes)
    samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    # Ensure mono
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


async def push_audio_to_source(
    audio_source: rtc.AudioSource,
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    """Push PCM audio samples into a LiveKit AudioSource track.

    LiveKit expects int16 PCM frames.  We chunk into ~10 ms frames to keep
    the stream smooth.

    Args:
        audio_source: The LiveKit AudioSource attached to our local track.
        samples: Float32 mono audio samples (range -1.0 … 1.0).
        sample_rate: Sample rate of *samples* (e.g. 24000).
    """
    # Convert float32 → int16
    pcm_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)

    # 10 ms chunks
    chunk_size = sample_rate // 100  # 10 ms worth of samples
    for start in range(0, len(pcm_int16), chunk_size):
        chunk = pcm_int16[start : start + chunk_size]
        if len(chunk) == 0:
            break
        frame = rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=sample_rate,
            num_channels=CHANNELS,
            samples_per_channel=len(chunk),
        )
        await audio_source.capture_frame(frame)
    # Small silence at the end so the last syllable isn't clipped
    silence = np.zeros(chunk_size * 5, dtype=np.int16)
    await audio_source.capture_frame(
        rtc.AudioFrame(
            data=silence.tobytes(),
            sample_rate=sample_rate,
            num_channels=CHANNELS,
            samples_per_channel=len(silence),
        )
    )


# ── Phone number extraction ───────────────────────────────────────────────────

def extract_phone(participant: rtc.RemoteParticipant) -> str:
    """Try to get the caller's phone number from participant metadata.

    LiveKit SIP bridges typically set the participant identity or a metadata
    field to the caller's E.164 number.  If none is found we fall back to
    the participant's SID so memory still works (just not cross-call).

    Args:
        participant: The remote participant representing the caller.

    Returns:
        A string key suitable for use with memory.load / memory.save.
    """
    # SIP bridge sets identity to the phone number in many configurations
    identity = participant.identity or ""
    if identity.startswith("+") or identity.lstrip("+").isdigit():
        return identity

    # Try JSON metadata
    import json as _json
    try:
        meta = _json.loads(participant.metadata or "{}")
        for key in ("phone_number", "phone", "caller_id", "from"):
            if key in meta:
                return str(meta[key])
    except (_json.JSONDecodeError, TypeError):
        pass

    # Fallback — use SID (unique per session, not cross-call)
    logger.warning(
        "Could not extract phone number from participant %s — using SID.",
        participant.sid,
    )
    return participant.sid


# ── Core conversation turn ────────────────────────────────────────────────────

async def handle_turn(
    transcript: str,
    phone: str,
    is_first_turn: bool,
    audio_source: rtc.AudioSource,
    http_client: httpx.AsyncClient,
) -> bool:
    """Process one conversation turn: STT transcript → LLM → TTS → audio out.

    Args:
        transcript: The caller's transcribed speech for this turn.
        phone: Caller's phone number / memory key.
        is_first_turn: True when this is the opening message of the call.
        audio_source: LiveKit AudioSource to push synthesised speech into.
        http_client: Shared async HTTP client.

    Returns:
        ``True`` if the agent should escalate to a human agent, else ``False``.
    """
    # ── Load history ──────────────────────────────────────────────────────────
    history = memory.load(phone)

    # ── Build LLM messages ────────────────────────────────────────────────────
    messages = prompts.build_messages(
        caller_history=history,
        current_user_message=transcript,
        is_first_turn=is_first_turn,
    )

    # ── Call LLM ─────────────────────────────────────────────────────────────
    try:
        raw_reply = await call_llm(messages, http_client)
    except RuntimeError:
        raw_reply = prompts.LLM_FAILURE

    # ── Escalation check ──────────────────────────────────────────────────────
    should_escalate = prompts.check_escalation(raw_reply)
    spoken_reply = prompts.clean_reply(raw_reply)

    if should_escalate:
        # Override with the canned hold message; the [ESCALATE] body may be
        # redundant or confusing to callers.
        spoken_reply = prompts.ESCALATION_HOLD

    logger.info("LINGO → %r", spoken_reply[:120])

    # ── Synthesise speech ─────────────────────────────────────────────────────
    try:
        wav_bytes = await call_tts(spoken_reply, http_client)
        samples, sr = wav_to_frames(wav_bytes)
        await push_audio_to_source(audio_source, samples, sr)
    except RuntimeError:
        # TTS failed — log and try to speak the canned error text
        logger.error("TTS failed for reply; attempting fallback TTS.")
        try:
            wav_bytes = await call_tts(prompts.TTS_FAILURE, http_client)
            samples, sr = wav_to_frames(wav_bytes)
            await push_audio_to_source(audio_source, samples, sr)
        except RuntimeError:
            logger.critical("Fallback TTS also failed — caller will hear silence.")

    # ── Persist history ───────────────────────────────────────────────────────
    # Only persist if the LLM actually replied (not a canned error)
    if spoken_reply not in (prompts.LLM_FAILURE, prompts.TTS_FAILURE, prompts.ESCALATION_HOLD):
        updated_history = history + [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": spoken_reply},
        ]
        memory.save(phone, updated_history)

    return should_escalate


# ── LiveKit job entrypoint ────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    """Main job entrypoint called by the LiveKit worker for each inbound call.

    The job lifecycle:
      1. Connect to the LiveKit room.
      2. Publish a local audio track (for LINGO's voice).
      3. Subscribe to the caller's audio track.
      4. Stream STT via Deepgram; process each final transcript.
      5. On call end / disconnect, clean up gracefully.
    """
    logger.info("New job — room: %s", ctx.room.name)

    # ── Connect to room ───────────────────────────────────────────────────────
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # ── Create outgoing audio track ───────────────────────────────────────────
    audio_source = rtc.AudioSource(
        sample_rate=SAMPLE_RATE,
        num_channels=CHANNELS,
    )
    local_track = rtc.LocalAudioTrack.create_audio_track(
        "lingo-voice", audio_source
    )
    publish_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await ctx.room.local_participant.publish_track(local_track, publish_opts)
    logger.info("Audio track published.")

    # ── Wait for the caller to join ───────────────────────────────────────────
    caller: Optional[rtc.RemoteParticipant] = None

    # Check existing participants first (race condition on fast joins)
    for p in ctx.room.remote_participants.values():
        caller = p
        break

    if caller is None:
        # Wait up to 30 s for the caller to appear
        participant_joined = asyncio.Event()

        @ctx.room.on("participant_connected")
        def _on_participant(participant: rtc.RemoteParticipant) -> None:
            nonlocal caller
            caller = participant
            participant_joined.set()

        try:
            await asyncio.wait_for(participant_joined.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("No caller joined within 30 s — closing job.")
            return

    phone = extract_phone(caller)
    logger.info("Caller identified as %s", phone)

    # ── STT setup ─────────────────────────────────────────────────────────────
    stt = deepgram.STT(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        language="en-US",
        detect_language=True,   # auto-detect if caller switches language
        interim_results=False,  # only fire on final transcripts
        punctuate=True,
        smart_format=True,
    )

    # ── Shared HTTP client (keep-alive for LLM + TTS calls) ───────────────────
    async with httpx.AsyncClient() as http_client:

        # ── Opening greeting ──────────────────────────────────────────────────
        # LINGO speaks first — no STT needed for the greeting turn.
        logger.info("Sending opening greeting to %s", phone)
        await handle_turn(
            transcript=prompts.GREETING,   # synthetic "caller said hi" trigger
            phone=phone,
            is_first_turn=True,
            audio_source=audio_source,
            http_client=http_client,
        )

        # ── STT stream loop ───────────────────────────────────────────────────
        # Subscribe to the caller's audio track for STT.
        caller_track: Optional[rtc.RemoteAudioTrack] = None
        for pub in caller.track_publications.values():
            if pub.track and isinstance(pub.track, rtc.RemoteAudioTrack):
                caller_track = pub.track
                break

        if caller_track is None:
            # Wait for the caller's track to be published
            track_available = asyncio.Event()

            @ctx.room.on("track_subscribed")
            def _on_track(
                track: rtc.Track,
                pub: rtc.TrackPublication,
                participant: rtc.RemoteParticipant,
            ) -> None:
                nonlocal caller_track
                if isinstance(track, rtc.RemoteAudioTrack):
                    caller_track = track
                    track_available.set()

            try:
                await asyncio.wait_for(track_available.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                logger.warning("Caller's audio track never appeared — ending call.")
                return

        logger.info("Streaming STT from caller's audio track.")

        turn_number = 0

        async with stt.stream() as stt_stream:
            # Feed caller audio into the STT stream
            async def _feed_audio() -> None:
                audio_stream = rtc.AudioStream(caller_track)
                async for event in audio_stream:
                    if isinstance(event, rtc.AudioFrameEvent):
                        await stt_stream.push_frame(event.frame)

            feed_task = asyncio.create_task(_feed_audio())

            try:
                async for stt_event in stt_stream:
                    # Only process final (non-interim) transcripts
                    if not stt_event.is_final:
                        continue

                    transcript = stt_event.alternatives[0].text.strip()

                    if not transcript:
                        # Deepgram returned an empty final result (e.g. silence)
                        continue

                    logger.info("Caller said: %r", transcript[:200])
                    turn_number += 1

                    # Pause audio feed while LINGO is speaking to avoid echo
                    feed_task.cancel()
                    try:
                        await feed_task
                    except asyncio.CancelledError:
                        pass

                    # Process the turn
                    escalate = await handle_turn(
                        transcript=transcript,
                        phone=phone,
                        is_first_turn=False,
                        audio_source=audio_source,
                        http_client=http_client,
                    )

                    if escalate:
                        logger.info("Escalating call for %s to human agent.", phone)
                        # In a production system you would trigger a SIP transfer here.
                        # For now we log the intent and end the bot session gracefully.
                        await _speak_canned(
                            prompts.GOODBYE, audio_source, http_client
                        )
                        break

                    # Resume feeding caller audio
                    feed_task = asyncio.create_task(_feed_audio())

            except Exception as exc:
                logger.exception("Unhandled error in STT loop: %s", exc)
                await _speak_canned(prompts.LLM_FAILURE, audio_source, http_client)
            finally:
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass

    logger.info("Call with %s ended.", phone)


async def _speak_canned(
    text: str,
    audio_source: rtc.AudioSource,
    http_client: httpx.AsyncClient,
) -> None:
    """Helper to speak a canned response without updating memory."""
    try:
        wav_bytes = await call_tts(text, http_client)
        samples, sr = wav_to_frames(wav_bytes)
        await push_audio_to_source(audio_source, samples, sr)
    except RuntimeError:
        logger.error("Could not synthesise canned message: %r", text[:80])


# ── Worker process setup ──────────────────────────────────────────────────────

def prewarm(proc: JobProcess) -> None:
    """Called once per worker process before any jobs are dispatched.

    Use this to load models or warm up connections that are expensive to
    initialise per-call.  Nothing heavy to preload right now, but the hook
    is here for future use (e.g. a local Whisper model).
    """
    logger.info("Worker process warmed up and ready.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _check_env()
    logger.info("Starting LINGO worker…")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
