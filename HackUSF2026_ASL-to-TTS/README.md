# ASL to TTS — HackUSF 2026

A real-time American Sign Language (ASL) fingerspelling translator that uses computer vision, a custom neural network, and AI to convert hand gestures into spoken words.

## How It Works

1. **Hand Tracking** — MediaPipe detects 21 hand landmarks from your webcam in real time
2. **Gesture Classification** — A PyTorch neural network trained on ASL landmark data classifies each gesture as a letter (A–Z)
3. **Live Vocalization** — In normal mode, each recognized letter is spoken aloud using pre-recorded audio
4. **Word Mode** — Press `W` to start recording a word. Sign each letter and hold it for ~0.7 seconds. Press `W` again to finalize
5. **AI Correction** — The recorded letters are sent to Google Gemini, which infers the closest real English word (correcting for any recognition errors)
6. **ElevenLabs TTS** — The corrected word is spoken aloud using ElevenLabs text-to-speech

## Controls

| Key | Action |
|-----|--------|
| `W` | Start / stop word recording |
| `Q` | Quit |

## Setup

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (required by ElevenLabs for audio playback)

```bash
# macOS
brew install ffmpeg
```

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd HackUSF2026_ASL-to-TTS

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

- **Gemini API key** — Get one at [aistudio.google.com](https://aistudio.google.com)
- **ElevenLabs API key** — Get one at [elevenlabs.io](https://elevenlabs.io)

### Run

```bash
venv/bin/python main.py
```

## Project Structure

```
├── main.py                  # Entry point — camera loop, W-key word mode, Gemini + ElevenLabs integration
├── src/
│   ├── tracker.py           # MediaPipe hand landmark detection and normalization
│   ├── neural_net.py        # PyTorch ASL classifier (trained at runtime)
│   └── vocalizer.py         # Pre-recorded letter audio playback with debounce
├── assets/
│   ├── audio/               # Pre-recorded .wav files for each letter (A–Z)
│   └── data/                # ASL landmark training CSV
└── .env                     # API keys (not committed)
```

## Tech Stack

- [MediaPipe](https://developers.google.com/mediapipe) — Hand landmark detection
- [PyTorch](https://pytorch.org/) — Neural network for gesture classification
- [Google Gemini](https://ai.google.dev/) — Word correction from fingerspelled letters
- [ElevenLabs](https://elevenlabs.io/) — Text-to-speech vocalization
- [OpenCV](https://opencv.org/) — Webcam capture and display
