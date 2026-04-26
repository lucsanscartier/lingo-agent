"""
agent.py — LINGO AI Phone Agent

Beta-hardened runtime:
  - LiveKit worker for inbound SIP/audio jobs
  - Deepgram streaming STT
  - Hugging Face router chat completions for LLM replies
  - Hugging Face hf-inference TTS endpoint with payload fallback
  - SQLite-backed caller memory via memory.py
  - FastAPI /health, /status, /metrics surface via health_server.py
"""

from __future__ import annotations

import asyncio
import io
import json as _json
import logging
import sys
from typing import Optional

import httpx
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import deepgram

from config import Settings
import health_server
import memory
import prompts

SETTINGS = Settings.from_env()

logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("lingo.agent")

SAMPLE_RATE = SETTINGS.sample_rate
CHANNELS = SETTINGS.channels


def _check_env() -> None:
    missing = SETTINGS.missing_required()
    if missing:
        logger.error(
            "Missing required environment variables: %s. Copy .env.example to .env "
            "locally or set these as Hugging Face Space secrets.",
            ", ".join(missing),
        )
        sys.exit(1)


def _hf_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {SETTINGS.hf_token}",
        "Content-Type": "application/json",
    }


async def call_llm(messages: list[dict], http_client: httpx.AsyncClient) -> str:
    """Call Hugging Face's OpenAI-compatible chat router."""
    payload = {
        "model": SETTINGS.hf_chat_model,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.5,
    }
    logger.debug("LLM request — model=%s messages=%d", SETTINGS.hf_chat_model, len(messages))
    try:
        resp = await http_client.post(
            SETTINGS.hf_chat_url,
            json=payload,
            headers=_hf_headers(),
            timeout=45.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        logger.debug("LLM reply — %d chars", len(text))
        return str(text).strip()
    except (httpx.HTTPStatusError, httpx.RequestError, KeyError, IndexError, TypeError) as exc:
        logger.error("LLM call failed: %s", exc)
        health_server.set_last_error(f"LLM failure: {exc}")
        raise RuntimeError("LLM failure") from exc


async def call_tts(text: str, http_client: httpx.AsyncClient) -> bytes:
    """Synthesize speech through Hugging Face hf-inference.

    Newer HF TTS examples use {"text_inputs": "..."} while some older
    model endpoints use {"inputs": "..."}. We try the modern form first,
    then fall back automatically for compatibility.
    """
    payloads = [{"text_inputs": text}, {"inputs": text}]
    last_exc: Exception | None = None

    for payload in payloads:
        try:
            resp = await http_client.post(
                SETTINGS.hf_tts_url,
                json=payload,
                headers=_hf_headers(),
                timeout=75.0,
            )
            if resp.status_code in {400, 404, 415, 422}:
                last_exc = RuntimeError(f"TTS rejected payload {list(payload.keys())}: {resp.text[:200]}")
                continue
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                raise RuntimeError(f"TTS returned JSON instead of audio: {resp.text[:300]}")
            return resp.content
        except (httpx.HTTPStatusError, httpx.RequestError, RuntimeError) as exc:
            logger.warning("TTS attempt failed: %s", exc)
            last_exc = exc

    health_server.set_last_error(f"TTS failure: {last_exc}")
    raise RuntimeError("TTS failure") from last_exc


def wav_to_frames(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to mono float32 samples."""
    buf = io.BytesIO(wav_bytes)
    samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


async def push_audio_to_source(
    audio_source: rtc.AudioSource,
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    """Push PCM audio samples into a LiveKit AudioSource track."""
    pcm_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    chunk_size = max(1, sample_rate // 100)  # ~10 ms chunks

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

    silence = np.zeros(chunk_size * 5, dtype=np.int16)
    await audio_source.capture_frame(
        rtc.AudioFrame(
            data=silence.tobytes(),
            sample_rate=sample_rate,
            num_channels=CHANNELS,
            samples_per_channel=len(silence),
        )
    )


def extract_phone(participant: rtc.RemoteParticipant) -> str:
    """Extract the caller phone/memory key from LiveKit participant data."""
    identity = participant.identity or ""
    if identity.startswith("+") or identity.lstrip("+").isdigit():
        return identity

    try:
        meta = _json.loads(participant.metadata or "{}")
        for key in ("phone_number", "phone", "caller_id", "from"):
            if key in meta:
                return str(meta[key])
    except (_json.JSONDecodeError, TypeError):
        pass

    logger.warning("Could not extract phone number from participant %s — using SID.", participant.sid)
    return participant.sid


async def speak_text(
    text: str,
    audio_source: rtc.AudioSource,
    http_client: httpx.AsyncClient,
) -> None:
    """Speak text without touching caller memory."""
    wav_bytes = await call_tts(text, http_client)
    samples, sr = wav_to_frames(wav_bytes)
    await push_audio_to_source(audio_source, samples, sr)


async def handle_turn(
    transcript: str,
    phone: str,
    is_first_turn: bool,
    audio_source: rtc.AudioSource,
    http_client: httpx.AsyncClient,
) -> bool:
    """Process one conversation turn: transcript → LLM → TTS → audio out."""
    history = memory.load(phone)
    messages = prompts.build_messages(
        caller_history=history,
        current_user_message=transcript,
        is_first_turn=is_first_turn,
    )

    try:
        raw_reply = await call_llm(messages, http_client)
    except RuntimeError:
        raw_reply = prompts.LLM_FAILURE

    should_escalate = prompts.check_escalation(raw_reply)
    spoken_reply = prompts.clean_reply(raw_reply)

    if should_escalate:
        spoken_reply = prompts.ESCALATION_HOLD
        health_server.record_escalation()
        memory.log_event(phone, "escalation_requested", {"transcript": transcript})

    logger.info("LINGO → %r", spoken_reply[:160])

    try:
        await speak_text(spoken_reply, audio_source, http_client)
    except RuntimeError:
        logger.error("TTS failed for reply; attempting fallback TTS.")
        try:
            await speak_text(prompts.TTS_FAILURE, audio_source, http_client)
        except RuntimeError:
            logger.critical("Fallback TTS also failed — caller will hear silence.")

    if spoken_reply not in (prompts.LLM_FAILURE, prompts.TTS_FAILURE, prompts.ESCALATION_HOLD):
        updated_history = history + [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": spoken_reply},
        ]
        memory.save(phone, updated_history)
        memory.log_event(phone, "turn_completed", {"chars_in": len(transcript), "chars_out": len(spoken_reply)})

    return should_escalate


async def _speak_canned(
    text: str,
    audio_source: rtc.AudioSource,
    http_client: httpx.AsyncClient,
) -> None:
    try:
        await speak_text(text, audio_source, http_client)
    except RuntimeError:
        logger.error("Could not synthesise canned message: %r", text[:80])


async def _notify_escalation(phone: str, transcript: str, http_client: httpx.AsyncClient) -> None:
    """Optional webhook hook for owner alerts or later SIP-transfer workflows."""
    if not SETTINGS.escalation_webhook_url:
        return

    payload = {
        "phone": phone,
        "reason": "caller requested human escalation",
        "last_transcript": transcript,
        "service": "lingo-agent",
    }
    try:
        resp = await http_client.post(
            SETTINGS.escalation_webhook_url,
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        memory.log_event(phone, "escalation_webhook_sent", {"status": resp.status_code})
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.error("Escalation webhook failed: %s", exc)
        health_server.set_last_error(f"Escalation webhook failed: {exc}")


async def entrypoint(ctx: JobContext) -> None:
    """Main LiveKit job entrypoint for each inbound call."""
    logger.info("New job — room: %s", ctx.room.name)
    health_server.record_call_start()

    phone = "unknown"
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        audio_source = rtc.AudioSource(sample_rate=SAMPLE_RATE, num_channels=CHANNELS)
        local_track = rtc.LocalAudioTrack.create_audio_track("lingo-voice", audio_source)
        publish_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await ctx.room.local_participant.publish_track(local_track, publish_opts)
        logger.info("Audio track published.")

        caller: Optional[rtc.RemoteParticipant] = None
        for p in ctx.room.remote_participants.values():
            caller = p
            break

        if caller is None:
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
        memory.log_event(phone, "call_started", {"room": ctx.room.name})

        stt = deepgram.STT(
            api_key=SETTINGS.deepgram_api_key,
            language="en-US",
            detect_language=True,
            interim_results=False,
            punctuate=True,
            smart_format=True,
        )

        async with httpx.AsyncClient() as http_client:
            logger.info("Sending opening greeting to %s", phone)
            await _speak_canned(prompts.GREETING, audio_source, http_client)

            caller_track: Optional[rtc.RemoteAudioTrack] = None
            for pub in caller.track_publications.values():
                if pub.track and isinstance(pub.track, rtc.RemoteAudioTrack):
                    caller_track = pub.track
                    break

            if caller_track is None:
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

            async with stt.stream() as stt_stream:
                async def _feed_audio() -> None:
                    audio_stream = rtc.AudioStream(caller_track)
                    async for event in audio_stream:
                        if isinstance(event, rtc.AudioFrameEvent):
                            await stt_stream.push_frame(event.frame)

                feed_task = asyncio.create_task(_feed_audio())

                try:
                    async for stt_event in stt_stream:
                        if not stt_event.is_final:
                            continue

                        transcript = stt_event.alternatives[0].text.strip()
                        if not transcript:
                            continue

                        logger.info("Caller said: %r", transcript[:200])

                        feed_task.cancel()
                        try:
                            await feed_task
                        except asyncio.CancelledError:
                            pass

                        escalate = await handle_turn(
                            transcript=transcript,
                            phone=phone,
                            is_first_turn=False,
                            audio_source=audio_source,
                            http_client=http_client,
                        )

                        if escalate:
                            logger.info("Escalating call for %s.", phone)
                            await _notify_escalation(phone, transcript, http_client)
                            await _speak_canned(prompts.GOODBYE, audio_source, http_client)
                            break

                        feed_task = asyncio.create_task(_feed_audio())

                except Exception as exc:
                    logger.exception("Unhandled error in STT loop: %s", exc)
                    health_server.set_last_error(f"STT loop error: {exc}")
                    await _speak_canned(prompts.LLM_FAILURE, audio_source, http_client)
                finally:
                    feed_task.cancel()
                    try:
                        await feed_task
                    except asyncio.CancelledError:
                        pass

    finally:
        memory.log_event(phone, "call_ended", {})
        health_server.record_call_end()
        logger.info("Call with %s ended.", phone)


def prewarm(proc: JobProcess) -> None:
    logger.info("Worker process warmed up and ready.")


if __name__ == "__main__":
    _check_env()
    if SETTINGS.health_enabled:
        health_server.start_health_server(SETTINGS.health_host, SETTINGS.health_port)
    health_server.mark_worker_started()
    logger.info("Starting LINGO worker…")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
