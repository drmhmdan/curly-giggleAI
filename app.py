"""
FastAPI backend server for speech transcription and AI response generation.
Uses faster-whisper for transcription and Google Gemini for AI responses.
"""
import logging
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("curly-giggleai")

app = FastAPI(title="Speech Transcription & AI Response")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache
whisper_models = {}
gemini_api_key = os.getenv("GEMINI_API_KEY", "")


def infer_upload_suffix(upload: UploadFile, default: str = ".webm") -> str:
    """Infer file suffix for an uploaded audio blob."""
    if upload and upload.filename:
        suffix = Path(upload.filename).suffix
        if suffix:
            return suffix

    content_type = getattr(upload, "content_type", None)
    if content_type:
        if content_type in {"audio/wav", "audio/x-wav"}:
            return ".wav"
        guessed = mimetypes.guess_extension(content_type)
        if guessed:
            return guessed

    return default

class TranscriptionRequest(BaseModel):
    model: str = "tiny"
    language: str = "de"

class GeminiRequest(BaseModel):
    text: str
    model: str = "gemini-2.0-flash-exp"

def get_whisper_model(model_name: str = "tiny"):
    """Get or create a Whisper model instance."""
    if WhisperModel is None:
        logger.error("faster-whisper import failed; transcription unavailable")
        raise HTTPException(status_code=500, detail="faster-whisper not installed")

    if model_name not in whisper_models:
        logger.info("Loading Whisper model '%s'", model_name)
        try:
            whisper_models[model_name] = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8"
            )
            logger.info("Whisper model '%s' loaded on CPU", model_name)
        except Exception as e:
            logger.exception("Failed to load Whisper model '%s'", model_name)
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    return whisper_models[model_name]

@app.post("/api/preload_model")
async def preload_model(model: str = Form("tiny")):
    """Preload a Whisper model to reduce first-use latency."""
    try:
        logger.info("Preloading Whisper model '%s'", model)
        get_whisper_model(model)
        return JSONResponse(content={
            "status": "success",
            "model": model,
            "message": f"Model '{model}' loaded successfully"
        })
    except Exception as e:
        logger.exception("Failed to preload model '%s'", model)
        raise HTTPException(status_code=500, detail=f"Failed to preload model: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        logger.debug("Serving static UI from %s", html_path)
        # Explicitly decode as UTF-8 to avoid Windows default cp1252 issues
        return html_path.read_text(encoding="utf-8")
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Speech Transcription</title></head>
    <body>
        <h1>Speech Transcription & AI Response</h1>
        <p>Static files not found. Please create static/index.html</p>
    </body>
    </html>
    """

@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    model: str = Form("tiny"),
    language: str = Form("de")
):
    """
    Transcribe audio file using faster-whisper.
    
    Args:
        audio: Audio file (webm, wav, mp3, etc.)
        model: Whisper model name (tiny, base, small, medium, large)
        language: Language code (e.g., 'de' for German, 'en' for English)
    
    Returns:
        JSON with transcription text
    """
    logger.info("Received transcription request: filename=%s model=%s language=%s", audio.filename, model, language)

    if WhisperModel is None:
        logger.error("Transcription requested but faster-whisper is not available")
        return JSONResponse(
            status_code=500,
            content={"error": "faster-whisper not installed. Install with: pip install faster-whisper"}
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=infer_upload_suffix(audio, ".webm")) as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        logger.debug("Audio saved to temporary path %s (%d bytes)", temp_audio_path, len(content))

        try:
            # Get Whisper model and transcribe
            whisper_model = get_whisper_model(model)
            logger.info("Starting transcription with model '%s'", model)
            segments, info = whisper_model.transcribe(
                temp_audio_path,
                language=language,
                beam_size=5
            )
            segments = list(segments)
            logger.info(
                "Transcription finished: language=%s prob=%.3f segments=%d",
                info.language,
                info.language_probability,
                len(segments)
            )

            # Combine all segments into full transcription
            transcription = " ".join([segment.text for segment in segments])
            logger.debug("Transcription text: %s", transcription.strip())
            return JSONResponse(content={
                "transcription": transcription.strip(),
                "language": info.language,
                "language_probability": info.language_probability
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                logger.debug("Removing temporary file %s", temp_audio_path)
                os.unlink(temp_audio_path)
    
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/transcribe_stream")
async def transcribe_audio_chunk(
    audio_chunk: UploadFile = File(...),
    model: str = Form("tiny"),
    language: str = Form("de")
):
    """Transcribe a small audio chunk for near real-time updates."""
    logger.debug(
        "Streaming chunk received: filename=%s size=%s model=%s language=%s",
        audio_chunk.filename,
        audio_chunk.size if hasattr(audio_chunk, "size") else "?",
        model,
        language,
    )

    if WhisperModel is None:
        logger.error("Chunk transcription requested but faster-whisper is not available")
        return JSONResponse(
            status_code=500,
            content={"error": "faster-whisper not installed. Install with: pip install faster-whisper"}
        )

    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=infer_upload_suffix(audio_chunk, ".webm")) as temp_audio:
            chunk_bytes = await audio_chunk.read()
            temp_audio.write(chunk_bytes)
            temp_audio_path = temp_audio.name
        logger.debug("Chunk stored at %s (%d bytes)", temp_audio_path, len(chunk_bytes))

        whisper_model = get_whisper_model(model)
        segments, info = whisper_model.transcribe(
            temp_audio_path,
            language=language,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            word_timestamps=False
        )
        segments = list(segments)
        text = " ".join(segment.text for segment in segments).strip()
        logger.debug(
            "Chunk transcription complete: language=%s len=%d", info.language, len(text)
        )

        return JSONResponse(content={
            "partial_transcription": text,
            "language": info.language,
            "language_probability": info.language_probability
        })

    except Exception as e:
        logger.exception("Chunk transcription failed")
        raise HTTPException(status_code=500, detail=f"Chunk transcription failed: {str(e)}")

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            logger.debug("Removing temporary chunk file %s", temp_audio_path)
            os.unlink(temp_audio_path)

@app.post("/api/gemini")
async def generate_gemini_response(request: GeminiRequest):
    """
    Generate AI response using Google Gemini.
    
    Args:
        request: Contains text prompt and model name
    
    Returns:
        JSON with AI-generated response
    """
    logger.info("Gemini request received for model '%s'", request.model)

    if genai is None:
        logger.error("Gemini SDK import failed")
        return JSONResponse(
            status_code=500,
            content={"error": "google-generativeai not installed. Install with: pip install google-generativeai"}
        )

    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable missing")
        return JSONResponse(
            status_code=500,
            content={"error": "GEMINI_API_KEY environment variable not set"}
        )

    try:
        genai.configure(api_key=gemini_api_key)
        
        # Map model names to actual Gemini model identifiers
        model_map = {
            "flash 2.5 (stable)": "gemini-2.0-flash-exp",
            "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
            "gemini-pro": "gemini-pro",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-1.5-flash": "gemini-1.5-flash"
        }
        
        model_name = model_map.get(request.model, "gemini-2.0-flash-exp")
        model = genai.GenerativeModel(model_name)
        
        logger.debug("Submitting prompt to Gemini model '%s'", model_name)
        response = model.generate_content(request.text)
        logger.info("Gemini response generated (%d chars)", len(response.text or ""))

        return JSONResponse(content={
            "response": response.text,
            "model": model_name
        })
    
    except Exception as e:
        logger.exception("Gemini API call failed")
        raise HTTPException(status_code=500, detail=f"Gemini API failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return JSONResponse(content={
        "status": "healthy",
        "whisper_available": WhisperModel is not None,
        "gemini_available": genai is not None and bool(gemini_api_key)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="debug")
