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
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    # Lazy import to avoid issues with PyAV on some systems
    import sys
    if 'faster_whisper' not in sys.modules:
        try:
            from faster_whisper import WhisperModel
        except (ImportError, Exception):
            WhisperModel = None
    else:
        from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("curly-giggleai")

app = FastAPI(title="Speech Transcription & AI Response")

# System instruction file paths
SYS_INSTRUCT_FILE = Path(__file__).parent / "sys_instruct"
SYS_INSTRUCT_FILE_2 = Path(__file__).parent / "sys_instruct2"
DEFAULT_SYS_INSTRUCT = "You are a helpful AI assistant."

# Global session variables for system instructions (in-memory, not persisted)
current_session_system_instruction: Optional[str] = None
current_session_system_instruction_2: Optional[str] = None

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

# Global session variable for Gemini API key (in-memory, not persisted to .env)
current_session_gemini_api_key: Optional[str] = None

def get_gemini_api_key() -> str:
    """Get Gemini API key from current session, or from environment."""
    global current_session_gemini_api_key
    if current_session_gemini_api_key is not None:
        return current_session_gemini_api_key
    return gemini_api_key


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


def get_system_instruction() -> str:
    """Get system instruction from current session, or file, or default."""
    # Check if there's a session-specific instruction first
    if current_session_system_instruction is not None:
        return current_session_system_instruction
    
    # Fall back to file
    if SYS_INSTRUCT_FILE.exists():
        try:
            return SYS_INSTRUCT_FILE.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning("Failed to read system instruction file: %s", e)
    return DEFAULT_SYS_INSTRUCT


def get_system_instruction_2() -> str:
    """Get second system instruction from current session, or file, or default."""
    # Check if there's a session-specific instruction first
    if current_session_system_instruction_2 is not None:
        return current_session_system_instruction_2
    
    # Fall back to file
    if SYS_INSTRUCT_FILE_2.exists():
        try:
            return SYS_INSTRUCT_FILE_2.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning("Failed to read system instruction file 2: %s", e)
    return DEFAULT_SYS_INSTRUCT


class TranscriptionRequest(BaseModel):
    model: str = Field(default="tiny", pattern="^(tiny|base|small|medium|large|large-v2|large-v3|large-v3-turbo|TheChola/whisper-large-v3-turbo-german-faster-whisper)$")
    language: str = Field(default="de", pattern="^(de|en|ar)$")

class GeminiRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model: str = Field(default="flash 2.5 (stable)")
    system_instruction: Optional[str] = Field(default=None)
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

def get_whisper_model(model_name: str = "tiny"):
    """Get or create a Whisper model instance."""
    if WhisperModel is None:
        logger.error("faster-whisper import failed; transcription unavailable")
        raise HTTPException(status_code=500, detail="faster-whisper not installed")

    if model_name not in whisper_models:
        logger.info("Loading Whisper model '%s'", model_name)
        try:
            # Set model cache directory
            model_cache_dir = Path(__file__).parent / ".cache" / "faster-whisper"
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Set HuggingFace API token in environment if available
            hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
                logger.info("HuggingFace token configured")
            
            # Try loading model with token if available, fallback to public access
            try:
                whisper_models[model_name] = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(model_cache_dir),
                    local_files_only=False  # Allow downloading from HuggingFace
                )
            except Exception as download_error:
                logger.warning("Model download with token failed, attempting without token: %s", str(download_error))
                # Clear token and try again as public access
                os.environ.pop("HF_TOKEN", None)
                os.environ["HF_TOKEN"] = ""
                whisper_models[model_name] = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(model_cache_dir),
                    local_files_only=False
                )
            logger.info("Whisper model '%s' loaded on CPU from %s", model_name, model_cache_dir)
        except Exception as e:
            logger.exception("Failed to load Whisper model '%s'", model_name)
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    return whisper_models[model_name]

@lru_cache(maxsize=5)
def get_gemini_model(model_name: str):
    """Get Gemini model name mapping."""
    logger.info("Getting Gemini model mapping for '%s'", model_name)
    # Map display names to actual model identifiers
    model_map = {
        "pro 2.5": "models/gemini-2.5-pro",
        "flash 2.5 (stable)": "models/gemini-2.5-flash",
        "flash 2.5 lite": "models/gemini-2.5-flash-lite"
    }
    return model_map.get(model_name, "models/gemini-2.5-flash")

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
                beam_size=1,
                best_of=1,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    threshold=0.5,
                    min_speech_duration_ms=100
                ),
                condition_on_previous_text=False,
                temperature=(0.0, 0.4),
                compression_ratio_threshold=1.5,
                no_speech_threshold=0.8,
                word_timestamps=False,
                repetition_penalty=2.0
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
            vad_parameters=dict(
                min_silence_duration_ms=300,
                threshold=0.5,
                min_speech_duration_ms=100
            ),
            condition_on_previous_text=False,
            temperature=(0.0, 0.4),
            compression_ratio_threshold=1.5,
            no_speech_threshold=0.8,
            word_timestamps=False,
            repetition_penalty=2.0
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
        request: Contains text prompt, model name, and optional system instruction
    
    Returns:
        JSON with AI-generated response
    """
    logger.info("Gemini request received for model '%s'", request.model)

    if genai is None or types is None:
        logger.error("Gemini SDK import failed")
        return JSONResponse(
            status_code=500,
            content={"error": "google-genai not installed. Install with: pip install google-genai"}
        )

    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable missing")
        return JSONResponse(
            status_code=500,
            content={"error": "GEMINI_API_KEY environment variable not set"}
        )

    try:
        # Get the mapped model name
        model_name = get_gemini_model(request.model)
        
        # Use provided system instruction or load from file
        system_instruction = request.system_instruction or get_system_instruction()
        
        # Get the current Gemini API key (session or environment)
        current_api_key = get_gemini_api_key()
        
        # Create client with API key
        client = genai.Client(api_key=current_api_key)
        
        logger.debug("Submitting prompt to Gemini model '%s' with system instruction", model_name)
        response = client.models.generate_content(
            model=model_name,
            contents=request.text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        logger.info("Gemini response generated (%d chars)", len(response.text or ""))

        return JSONResponse(content={
            "response": response.text,
            "model": model_name
        })
    
    except Exception as e:
        logger.exception("Gemini API call failed")
        raise HTTPException(status_code=500, detail=f"Gemini API failed: {str(e)}")


@app.post("/api/gemini_2")
async def generate_gemini_response_2(request: GeminiRequest):
    """
    Generate AI response using Google Gemini with second system instruction.
    
    Args:
        request: Contains text prompt, model name, and optional system instruction
    
    Returns:
        JSON with AI-generated response
    """
    logger.info("Gemini 2 request received for model '%s'", request.model)

    if genai is None or types is None:
        logger.error("Gemini SDK import failed")
        return JSONResponse(
            status_code=500,
            content={"error": "google-genai not installed. Install with: pip install google-genai"}
        )

    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable missing")
        return JSONResponse(
            status_code=500,
            content={"error": "GEMINI_API_KEY environment variable not set"}
        )

    try:
        # Get the mapped model name
        model_name = get_gemini_model(request.model)
        
        # Use provided system instruction or load second instruction from file
        system_instruction = request.system_instruction or get_system_instruction_2()
        
        # Get the current Gemini API key (session or environment)
        current_api_key = get_gemini_api_key()
        
        # Create client with API key
        client = genai.Client(api_key=current_api_key)
        
        logger.debug("Submitting prompt to Gemini 2 model '%s' with second system instruction", model_name)
        response = client.models.generate_content(
            model=model_name,
            contents=request.text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        logger.info("Gemini 2 response generated (%d chars)", len(response.text or ""))

        return JSONResponse(content={
            "response": response.text,
            "model": model_name
        })
    
    except Exception as e:
        logger.exception("Gemini 2 API call failed")
        raise HTTPException(status_code=500, detail=f"Gemini 2 API failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return JSONResponse(content={
        "status": "healthy",
        "whisper_available": WhisperModel is not None,
        "gemini_available": genai is not None and bool(gemini_api_key)
    })


@app.get("/api/system_instruction")
async def get_system_instruction_endpoint():
    """Get current system instruction (from session)."""
    try:
        instruction = get_system_instruction()
        return JSONResponse(content={
            "system_instruction": instruction
        })
    except Exception as e:
        logger.exception("Failed to get system instruction")
        raise HTTPException(status_code=500, detail=f"Failed to get system instruction: {str(e)}")


@app.post("/api/system_instruction")
async def save_system_instruction_endpoint(instruction: str = Form(...)):
    """Save system instruction for current session (in-memory only, not persisted to file)."""
    global current_session_system_instruction
    
    try:
        if not instruction.strip():
            raise HTTPException(status_code=400, detail="System instruction cannot be empty")
        
        # Store in session (global variable) - NOT to file
        current_session_system_instruction = instruction.strip()
        logger.info("System instruction updated for current session")
        
        return JSONResponse(content={
            "status": "success",
            "message": "System instruction updated for this session"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update system instruction")
        raise HTTPException(status_code=500, detail=f"Failed to update system instruction: {str(e)}")


@app.get("/api/system_instruction_2")
async def get_system_instruction_2_endpoint():
    """Get current second system instruction (from session)."""
    try:
        instruction = get_system_instruction_2()
        return JSONResponse(content={
            "system_instruction": instruction
        })
    except Exception as e:
        logger.exception("Failed to get second system instruction")
        raise HTTPException(status_code=500, detail=f"Failed to get second system instruction: {str(e)}")


@app.post("/api/system_instruction_2")
async def save_system_instruction_2_endpoint(instruction: str = Form(...)):
    """Save second system instruction for current session (in-memory only, not persisted to file)."""
    global current_session_system_instruction_2
    
    try:
        if not instruction.strip():
            raise HTTPException(status_code=400, detail="System instruction cannot be empty")
        
        # Store in session (global variable) - NOT to file
        current_session_system_instruction_2 = instruction.strip()
        logger.info("Second system instruction updated for current session")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Second system instruction updated for this session"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update second system instruction")
        raise HTTPException(status_code=500, detail=f"Failed to update second system instruction: {str(e)}")


@app.get("/api/gemini_api_key")
async def get_gemini_api_key_endpoint():
    """Get current Gemini API key (plain text)."""
    try:
        current_key = get_gemini_api_key()
        if current_key:
            return JSONResponse(content={
                "api_key": current_key,
                "has_key": True,
                "key_length": len(current_key)
            })
        else:
            return JSONResponse(content={
                "api_key": None,
                "has_key": False,
                "key_length": 0
            })
    except Exception as e:
        logger.exception("Failed to get Gemini API key")
        raise HTTPException(status_code=500, detail=f"Failed to get Gemini API key: {str(e)}")


@app.post("/api/gemini_api_key")
async def set_gemini_api_key_endpoint(api_key: str = Form(...)):
    """Set Gemini API key for current session (in-memory only, not persisted to .env)."""
    global current_session_gemini_api_key
    
    try:
        if not api_key.strip():
            raise HTTPException(status_code=400, detail="API key cannot be empty")
        
        # Store in session (global variable) - NOT to .env file
        current_session_gemini_api_key = api_key.strip()
        logger.info("Gemini API key updated for current session")
        
        return JSONResponse(content={
            "status": "success",
            "message": "Gemini API key updated for this session"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update Gemini API key")
        raise HTTPException(status_code=500, detail=f"Failed to update Gemini API key: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    
    # Get absolute paths to SSL certificates
    project_root = Path(__file__).parent
    ssl_keyfile = project_root / "key.pem"
    ssl_certfile = project_root / "cert.pem"
    
    if ssl_keyfile.exists() and ssl_certfile.exists():
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8031,
            log_level="info",
            ssl_keyfile=str(ssl_keyfile),
            ssl_certfile=str(ssl_certfile)
        )
    else:
        print(f"Error: SSL certificates not found!")
        print(f"  Key: {ssl_keyfile}")
        print(f"  Cert: {ssl_certfile}")
        print("Generate them with: openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj \"/CN=localhost\"")
        exit(1)
