#!/usr/bin/env python3
import io
import logging
import os
import time
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

VOICES_PATH = 'v1_assets/voice_styles'

# Initialize TTS once at import time
# First run will auto download the model (~260MB)
tts = TTS(model_dir='v1_assets', auto_download=False)

# You can tweak threads here if you want, eg:
# tts = TTS(intra_op_num_threads=4, inter_op_num_threads=4)


class SynthesizeRequest(BaseModel):
    input: str
    voice: str
    speed: float = 1.05
    total_steps: int = 10
    max_chunk_length: int = 300


@app.get("/voices")
def list_voices():
    """
    Return list of available voice style names.
    """
    voices = set()
    for file in os.listdir(VOICES_PATH):
        if file.endswith(".json"):
            voices.add(file[:-5])

    return {"voices": sorted(list(voices))}

    # return {"voices": list(tts.voice_style_names)}


def ndarray_to_bytes(wav: np.ndarray, sample_rate: int = 44100) -> bytes:
    """
    Convert output (1, num_samples) float array to 16 bit WAV bytes
    and return the duration in seconds.
    """
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav[0]
    elif wav.ndim != 1:
        raise ValueError(f"Unexpected wav shape: {wav.shape}")

    wav = np.clip(wav, -1.0, 1.0)
    wav_int16 = (wav * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bit
        wf.setframerate(sample_rate)
        wf.writeframes(wav_int16.tobytes())

    return buf.getvalue()


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
    }
    """
    try:
        style = tts.get_voice_style(req.voice)
    except Exception as e:
        logger.warning("Invalid voice style %r: %s", req.voice, e)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{req.voice}'. Try one of /voices.",
        )
    print(f"{req.voice} {req.speed}➡️{req.input}⬅️")
    t0 = time.perf_counter()

    try:
        wav, [duration] = tts.synthesize(
            req.input,
            voice_style=style,
            total_steps=req.total_steps,
            speed=req.speed,
            silence_duration=0.25,
            max_chunk_length=req.max_chunk_length,
        )
    except Exception as e:
        logger.exception("Supertonic synthesis failed")
        raise HTTPException(status_code=500, detail="Synthesis failed")
    wav_bytes = ndarray_to_bytes(wav)
    elapsed = time.perf_counter() - t0
    rtf = elapsed / duration if duration > 0 else 0
    print(f"[{elapsed:.3f}s] len={len(req.input)} dur={duration:.2f}s rtf={rtf:.4f}")

    return Response(content=wav_bytes, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
    )