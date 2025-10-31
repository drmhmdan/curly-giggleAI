# curly-giggleAI

A modern web-based speech transcription and AI response application. Record audio directly from your browser, get accurate transcriptions using faster-whisper, and receive intelligent AI responses powered by Google Gemini.

## Features

- 🎙️ **Browser-based Audio Recording** - Record audio directly from your web browser
- 📝 **Accurate Transcription** - Uses faster-whisper for high-quality speech-to-text
- 🤖 **AI Responses** - Get intelligent responses from Google Gemini AI
- 🌍 **Multi-language Support** - Transcribe in German, English, Spanish, French, and more
- ⚙️ **Configurable Models** - Choose from different Whisper and Gemini models
- 🎨 **Modern UI/UX** - Clean, responsive design following 2025 best practices
- ⚡ **Real-time Processing** - Automatic transcription and AI response generation

## UI Elements

### Configuration Panel
- **Whisper Model**: Select from tiny, base, small, medium, or large models
- **Transcription Language**: Choose your preferred language (default: German)
- **Gemini Model**: Select AI model (default: flash 2.5 stable)
- **Auto-send after stop**: Automatically process audio when recording stops

### Recording Controls
- Start/Stop recording buttons
- Real-time status indicator
- Visual feedback during recording and processing

### Output Display
- **Transcription**: Real-time display of transcribed text
- **Gemini Response**: AI-generated responses based on transcription

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Google Gemini API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/drmhmdan/curly-giggleAI.git
cd curly-giggleAI
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
GEMINI_API_KEY=your-api-key-here
```

## Usage

### Quick Start (Easiest)
```bash
./start.sh
```

The startup script will check dependencies and start the server automatically.

### Manual Start

**Option 1: Using Python directly**
```bash
python app.py
```

**Option 2: Using uvicorn**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Option 3: Using Docker**
```bash
docker-compose up
```

### Access the Application

1. Open your browser and navigate to:
```
http://localhost:8000
```

2. Allow microphone permissions when prompted

3. Configure your settings in the left panel:
   - Select Whisper model (base recommended for balance of speed/accuracy)
   - Choose transcription language
   - Select Gemini model
   - Enable/disable auto-send

4. Click "Start Recording" to begin recording audio

5. Click "Stop Recording" when finished

6. View the transcription and AI response in the output panels

For a detailed quick start guide, see [QUICKSTART.md](QUICKSTART.md).

## API Endpoints

### `POST /api/transcribe`
Transcribe audio file using faster-whisper.

**Parameters:**
- `audio`: Audio file (webm, wav, mp3)
- `model`: Whisper model name (tiny, base, small, medium, large)
- `language`: Language code (e.g., 'de', 'en')

**Response:**
```json
{
  "transcription": "transcribed text",
  "language": "de",
  "language_probability": 0.99
}
```

### `POST /api/gemini`
Generate AI response using Google Gemini.

**Request Body:**
```json
{
  "text": "input text",
  "model": "flash 2.5 (stable)"
}
```

**Response:**
```json
{
  "response": "AI-generated response",
  "model": "gemini-2.0-flash-exp"
}
```

### `GET /api/health`
Check service health and availability.

**Response:**
```json
{
  "status": "healthy",
  "whisper_available": true,
  "gemini_available": true
}
```

## Docker Deployment

For easy deployment with Docker:

1. Build and run with docker-compose:
```bash
export GEMINI_API_KEY="your-api-key-here"
docker-compose up -d
```

2. Or build and run manually:
```bash
docker build -t curly-giggle-ai .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-api-key-here" curly-giggle-ai
```

3. Access the application at `http://localhost:8000`

## Technology Stack

- **Backend**: FastAPI (Python)
- **Transcription**: faster-whisper
- **AI**: Google Gemini API
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Audio**: MediaRecorder API
- **Deployment**: Docker (optional)

## Browser Compatibility

- Chrome/Edge 49+
- Firefox 25+
- Safari 14+
- Opera 36+

## Troubleshooting

### Microphone not working
- Ensure your browser has microphone permissions
- Check that no other application is using the microphone
- Try using HTTPS (required for some browsers)

### Transcription errors
- Check that faster-whisper is properly installed
- Try a smaller model (tiny or base) if having memory issues
- Ensure audio quality is good

### Gemini API errors
- Verify your GEMINI_API_KEY is set correctly
- Check your API quota and limits
- Ensure you have internet connectivity

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.