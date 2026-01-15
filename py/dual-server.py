#!/usr/bin/env python3
import io
import logging
import os
import time
import wave
import numpy as np
import v1_helper as st
import helper as st2

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel


logger = logging.getLogger("supertonic")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Supertonic TTS Server")
tts = st.load_text_to_speech("v1_assets/onnx", use_gpu=False)
tts2 = st2.load_text_to_speech("assets/onnx", use_gpu=False)
VOICES_PATH = "assets/voice_styles"


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
    for file in os.listdir(f"v1_{VOICES_PATH}"):
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
        # use supertonic v1 for English, v2 for other languages
        if req.lang == 'en':
            style = st.load_voice_style([f"v1_{VOICES_PATH}/{req.voice}.json"], verbose=False)
        else:
            req.speed = 1.1
            style = st2.load_voice_style([f"{VOICES_PATH}/{req.voice}.json"], verbose=False)
    except Exception as e:
        logger.warning("Invalid voice style %r: %s", req.voice, e)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{req.voice}'. Try one of /voices.",
        )
    print(f"{req.voice} {req.speed} {req.lang}➡️{req.input}⬅️")

    params = [req.input, style, req.total_steps, req.speed]
    if req.lang != 'en':
        params.insert(1, req.lang)

    t0 = time.perf_counter()
    try:
        wav, [duration] = tts(*params) if req.lang == 'en' else tts2(*params)
    except Exception as e:
        logger.exception("Supertonic synthesis failed {e}")
        raise HTTPException(status_code=500, detail="Synthesis failed")
    wav_bytes = ndarray_to_bytes(wav)
    elapsed = time.perf_counter() - t0
    rtf = elapsed / duration if duration > 0 else 0
    print(f"[{elapsed:.3f}s] len={len(req.input)} dur={duration:.2f}s rtf={rtf:.4f}")

    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/stream")
@app.post("/v1/audio/speech/stream")
async def stream_speech(req: SynthesizeRequest):
    """
    Stream speech chunks as raw PCM16 bytes.
    """
    try:
        if req.lang == 'en':
            style = st.load_voice_style([f"v1_{VOICES_PATH}/{req.voice}.json"], verbose=False)
            stream_func = tts.stream
        else:
            req.speed = 1.1
            style = st2.load_voice_style([f"{VOICES_PATH}/{req.voice}.json"], verbose=False)
            stream_func = tts2.stream
    except Exception as e:
        logger.warning("Invalid voice style %r: %s", req.voice, e)
        raise HTTPException(status_code=400, detail=f"Unknown voice '{req.voice}'")

    params = [req.input, style, req.total_steps, req.speed]
    if req.lang != 'en':
        params.insert(1, req.lang)

    def audio_generator():
        try:
            for chunk, _ in stream_func(*params):
                # Convert float32 to int16 PCM
                if chunk.ndim == 2:
                    chunk = chunk[0]
                chunk = np.clip(chunk, -1.0, 1.0)
                pcm_bytes = (chunk * 32767.0).astype(np.int16).tobytes()
                yield pcm_bytes
        except Exception as e:
            logger.exception("Streaming failed {e}")

    # media_type audio/l16 represents Linear 16-bit PCM
    return StreamingResponse(audio_generator(), media_type="audio/l16; rate=44100")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
    )