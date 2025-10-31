# Application Architecture

This document describes the architecture and design decisions of the Speech Transcription & AI Response application.

## Overview

The application follows a client-server architecture with a Python backend and a JavaScript frontend, designed for simplicity, security, and ease of deployment.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Frontend (HTML/CSS/JS)                     │ │
│  │  - Audio Recording (MediaRecorder API)                 │ │
│  │  - UI Components (Configuration, Controls, Display)    │ │
│  │  - API Communication (Fetch)                           │ │
│  └─────────────────────┬──────────────────────────────────┘ │
└────────────────────────┼────────────────────────────────────┘
                         │ HTTP/REST API
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Backend (Python/FastAPI)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FastAPI Application Server                           │  │
│  │  - CORS Middleware                                    │  │
│  │  - Request Validation (Pydantic)                      │  │
│  │  - Error Handling                                     │  │
│  └───────┬────────────────────────┬─────────────────────┘  │
│          │                        │                          │
│  ┌───────▼──────────┐    ┌───────▼─────────────┐          │
│  │  Transcription   │    │   AI Response        │          │
│  │  Service         │    │   Service            │          │
│  │  (faster-whisper)│    │   (Google Gemini)    │          │
│  └──────────────────┘    └──────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Frontend (static/index.html)

**Technology Stack:**
- Vanilla JavaScript (no frameworks for simplicity)
- Modern CSS with CSS Variables
- HTML5 MediaRecorder API

**Key Features:**
1. **Audio Recording**
   - Uses browser's MediaRecorder API
   - Captures audio in WebM format
   - Automatic stream cleanup after recording

2. **Configuration Panel**
   - Dropdowns for model selection
   - Language selector with 10+ languages
   - Checkbox for auto-send behavior

3. **Real-time Status**
   - Visual indicators (colored dots)
   - Status messages
   - Button state management

4. **Responsive Design**
   - Grid-based layout
   - Mobile-friendly
   - Dark theme for reduced eye strain

### Backend (app.py)

**Technology Stack:**
- FastAPI (async web framework)
- faster-whisper (speech-to-text)
- Google Generative AI (Gemini)
- Pydantic (data validation)

**Endpoints:**

1. **GET /**
   - Serves the main HTML page
   - Fallback if static files are not found

2. **POST /api/transcribe**
   - Accepts audio file (multipart/form-data)
   - Parameters: model, language
   - Returns: transcription text, detected language, confidence
   - Handles: temporary file creation/cleanup

3. **POST /api/gemini**
   - Accepts JSON with text and model
   - Returns: AI-generated response
   - Handles: API key validation, error responses

4. **GET /api/health**
   - Returns service status
   - Checks: dependency availability, API key presence

**Design Patterns:**

1. **Graceful Degradation**
   - Dependencies are optional at import time
   - Clear error messages if services unavailable
   - Health check endpoint for monitoring

2. **Resource Management**
   - Temporary files automatically cleaned up
   - Audio streams properly closed
   - Model caching to reduce memory usage

3. **Security**
   - CORS properly configured
   - API keys from environment variables
   - Input validation with Pydantic
   - Temporary file isolation

## Data Flow

### Transcription Flow

```
User clicks "Start Recording"
         ↓
Browser requests microphone access
         ↓
MediaRecorder captures audio
         ↓
User clicks "Stop Recording"
         ↓
Audio blob created (WebM format)
         ↓
FormData sent to /api/transcribe
         ↓
Backend saves to temporary file
         ↓
faster-whisper processes audio
         ↓
Transcription returned to frontend
         ↓
Text displayed in transcription field
         ↓
Temporary file deleted
```

### AI Response Flow

```
Transcription completed
         ↓
(If auto-send enabled)
         ↓
Text sent to /api/gemini
         ↓
Backend validates API key
         ↓
Request sent to Gemini API
         ↓
Response received
         ↓
Text returned to frontend
         ↓
Displayed in Gemini response field
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Google Gemini API key (required for AI features)

### Model Options

**Whisper Models:**
- `tiny`: Fastest, least accurate (~1GB RAM)
- `base`: Balanced (recommended, ~1GB RAM)
- `small`: Better accuracy (~2GB RAM)
- `medium`: High accuracy (~5GB RAM)
- `large`: Best accuracy (~10GB RAM)

**Gemini Models:**
- `flash 2.5 (stable)`: Default, fast responses
- `gemini-1.5-pro`: More capable, slower
- `gemini-1.5-flash`: Fast, efficient
- `gemini-pro`: General purpose

## Deployment Options

### 1. Direct Python Execution
Best for development and testing.

```bash
python app.py
```

### 2. Uvicorn (Production)
Recommended for production with process management.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Docker
Best for containerized deployments.

```bash
docker-compose up -d
```

### 4. Startup Script
Easiest for end users.

```bash
./start.sh
```

## Performance Considerations

1. **Whisper Model Selection**
   - Larger models = better accuracy but slower
   - Choose based on available RAM and latency requirements

2. **Caching**
   - Whisper models cached after first load
   - Reduces initialization time for subsequent requests

3. **Async/Await**
   - FastAPI uses async handlers
   - Non-blocking I/O operations
   - Better concurrency

4. **Resource Cleanup**
   - Temporary files deleted immediately
   - Memory efficient for long-running deployments

## Security Considerations

1. **API Key Management**
   - Keys stored in environment variables
   - Never committed to source control
   - Validated before use

2. **File Upload Security**
   - Files saved to secure temporary directory
   - Automatic cleanup after processing
   - No persistent storage of user audio

3. **CORS Configuration**
   - Currently allows all origins (development)
   - Should be restricted in production

4. **Input Validation**
   - Pydantic models validate all inputs
   - Type checking and constraints
   - Error messages don't leak sensitive info

## Future Enhancements

Potential improvements for future versions:

1. **Authentication**
   - User accounts
   - API key management
   - Usage tracking

2. **Storage**
   - Optional audio/transcription storage
   - History/search functionality
   - Export capabilities

3. **Real-time Processing**
   - WebSocket support
   - Streaming transcription
   - Progressive responses

4. **Additional Models**
   - Support for other transcription services
   - Multiple AI backends
   - Custom model fine-tuning

5. **Advanced Features**
   - Speaker diarization
   - Timestamp annotations
   - Confidence scores display
   - Audio waveform visualization

## Troubleshooting

### Common Issues

1. **Microphone Access Denied**
   - Check browser permissions
   - Use HTTPS for production
   - Verify no other app using microphone

2. **Transcription Errors**
   - Check faster-whisper installation
   - Verify ffmpeg is installed
   - Try smaller model if out of memory

3. **Gemini API Errors**
   - Verify API key is set
   - Check API quota/limits
   - Ensure internet connectivity

4. **Port Already in Use**
   - Change port in app.py
   - Kill existing process
   - Use different port with uvicorn

## Contributing

When contributing to this project:

1. Maintain the clean architecture
2. Add tests for new features
3. Update documentation
4. Follow existing code style
5. Ensure security best practices

## License

MIT License - See LICENSE file for details.
