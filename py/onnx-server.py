#!/usr/bin/env python3
import io
import logging
import os
import sys
import time
import wave
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


logger = logging.getLogger("supertonic")
logging.basicConfig(level=logging.INFO)

v2 = len(sys.argv) > 1 and (sys.argv[1] == '--v2')

if v2:
    from helper import load_text_to_speech, load_voice_style
    prefix = ''
else:
    prefix = 'v1_'
    from v1_helper import load_text_to_speech, load_voice_style

app = FastAPI(title="Supertonic TTS Server")
tts = load_text_to_speech(f"{prefix}assets/onnx", use_gpu=False)
VOICES_PATH = f"{prefix}assets/voice_styles"


class SynthesizeRequest(BaseModel):
    input: str
    voice: str
    speed: float = 1.2
    lang: str = 'en'
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
      "total_steps": 10,
      "max_chunk_length": 300,
    }
    """
    try:
        style = load_voice_style([f"{VOICES_PATH}/{req.voice}.json"], verbose=False)
    except Exception as e:
        logger.warning("Invalid voice style %r: %s", req.voice, e)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{req.voice}'. Try one of /voices.",
        )
    print(f"{req.voice} {req.speed} {req.lang}➡️{req.input}⬅️")

    params = [req.input, style, req.total_steps, req.speed]
    if v2:
        params.insert(1, req.lang)

    t0 = time.perf_counter()
    try:
        wav, [duration] = tts(*params)
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