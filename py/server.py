#!/usr/bin/env python3
import io
import logging
import subprocess
import wave
from typing import Literal, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from supertonic import TTS

logger = logging.getLogger("supertonic_server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Supertonic TTS Server")

# Initialize TTS once at import time
# First run will auto download the model (~260MB)
tts = TTS(auto_download=True)

# You can tweak threads here if you want, eg:
# tts = TTS(intra_op_num_threads=4, inter_op_num_threads=4)


class SynthesizeRequest(BaseModel):
    input: str
    voice: str
    speed: float = 1.05
    total_steps: int = 5
    max_chunk_length: int = 300
    response_format: Literal["wav", "ogg"] = "ogg"


@app.get("/voices")
def list_voices():
    """
    Return list of available voice style names.
    """
    return {"voices": list(tts.voice_style_names)}


def wav_ndarray_to_bytes(wav: np.ndarray, sample_rate: int = 44100) -> bytes:
    """
    Convert Supertonic output (1, num_samples) float array to 16 bit WAV bytes.

    Note: Supertonic outputs 16 bit WAV in its examples, and a sample rate
    around 44100 Hz is typical for models like this. Adjust sample_rate if
    Supertonic exposes a specific rate in the future.
    """
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav[0]
    elif wav.ndim != 1:
        raise ValueError(f"Unexpected wav shape: {wav.shape}")

    # Clip to [-1, 1] and convert to int16 PCM
    wav = np.clip(wav, -1.0, 1.0)
    wav_int16 = (wav * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bit
        wf.setframerate(sample_rate)
        wf.writeframes(wav_int16.tobytes())

    return buf.getvalue()


def wav_to_ogg_bytes(wav_bytes: bytes) -> bytes:
    """
    Convert WAV bytes to OGG bytes using ffmpeg entirely in memory.
    Requires ffmpeg to be installed and on PATH.
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-f",
                "ogg",
                "pipe:1",
            ],
            input=wav_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        logger.exception("ffmpeg not found")
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found on server PATH",
        )

    if proc.returncode != 0:
        logger.error("ffmpeg error: %s", proc.stderr.decode("utf-8", "ignore"))
        raise HTTPException(
            status_code=500,
            detail="ffmpeg failed to convert WAV to OGG",
        )

    return proc.stdout


@app.post("/cancel")
@app.post("/v1/audio/speech/cancel")
async def cancel():
    return {'ok': True}


@app.post("/synthesize")
@app.post("/v1/audio/speech")
def synthesize(req: SynthesizeRequest):
    """
    Synthesize speech from text using Supertonic.

    Body JSON:
    {
      "input": "text to synthesize",
      "voice": "F1",
      "speed": 1.05,
      "total_steps": 5,
      "max_chunk_length": 300,
      "response_format": "ogg" | "wav"
    }
    """
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input text is empty")

    try:
        style = tts.get_voice_style(req.voice)
    except Exception as e:
        logger.warning("Invalid voice style %r: %s", req.voice, e)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{req.voice}'. Try one of /voices.",
        )

    try:
        wav, duration = tts.synthesize(
            req.input,
            voice_style=style,
            total_steps=req.total_steps,
            speed=req.speed,
            silence_duration=0.5,
            max_chunk_length=req.max_chunk_length,
        )
    except Exception as e:
        logger.exception("Supertonic synthesis failed")
        raise HTTPException(status_code=500, detail="Synthesis failed")

    wav_bytes = wav_ndarray_to_bytes(wav)

    if req.response_format == "wav":
        return Response(content=wav_bytes, media_type="audio/wav")

    # Default ogg
    ogg_bytes = wav_to_ogg_bytes(wav_bytes)
    return Response(content=ogg_bytes, media_type="audio/ogg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )