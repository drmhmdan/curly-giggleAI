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

1. Start the server:
```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Allow microphone permissions when prompted

4. Configure your settings in the left panel:
   - Select Whisper model (base recommended for balance of speed/accuracy)
   - Choose transcription language
   - Select Gemini model
   - Enable/disable auto-send

5. Click "Start Recording" to begin recording audio

6. Click "Stop Recording" when finished

7. View the transcription and AI response in the output panels

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

## Technology Stack

- **Backend**: FastAPI (Python)
- **Transcription**: faster-whisper
- **AI**: Google Gemini API
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Audio**: MediaRecorder API

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