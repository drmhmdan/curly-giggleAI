# Quick Start Guide

Get up and running with the Speech Transcription & AI Response app in minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- Modern web browser (Chrome, Firefox, Safari, or Edge)
- Microphone access

## Installation (3 Steps)

### 1. Clone the repository

```bash
git clone https://github.com/drmhmdan/curly-giggleAI.git
cd curly-giggleAI
```

### 2. Set your Gemini API key

**Option A: Environment variable (recommended)**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Option B: Create a .env file**
```bash
echo 'GEMINI_API_KEY=your-api-key-here' > .env
```

### 3. Run the startup script

**On Linux/macOS:**
```bash
./start.sh
```

**On Windows:**
```bash
python app.py
```

The script will automatically:
- Check for Python installation
- Install required dependencies
- Verify your API key
- Start the server

## Usage

1. Open your browser and go to: `http://localhost:8000`

2. Allow microphone permissions when prompted

3. Configure your settings in the left panel:
   - **Whisper Model**: Choose transcription quality (base recommended)
   - **Language**: Select your spoken language (default: German)
   - **Gemini Model**: Choose AI model (default: flash 2.5)
   - **Auto-send**: Enable to automatically process after recording

4. Click "Start Recording" and speak

5. Click "Stop Recording" when finished

6. View your transcription and AI response!

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### Microphone not working
- Check browser permissions
- Make sure no other app is using the microphone
- Try using HTTPS if on a remote server

### API key error
- Verify your `GEMINI_API_KEY` is set correctly
- Check that your API key is active
- Ensure you have API quota available

### Port 8000 already in use
Change the port in `app.py` or run:
```bash
uvicorn app:app --port 8080
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore different Whisper models for better accuracy
- Try different languages
- Customize the prompt sent to Gemini

## Support

If you encounter any issues, please:
1. Check the [README.md](README.md) troubleshooting section
2. Review the console output for error messages
3. Open an issue on GitHub with details

Enjoy transcribing! 🎙️✨
