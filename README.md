# Voice-Based Authentication System

A biometric voice authentication and user management system built with Flask, PyTorch CNNs, Librosa acoustic feature extraction, and SQLite.

---

## Architecture Overview

```
voice-auth/
├── app.py               # Flask entry point and application factory
├── database.py          # SQLite database schema, user CRUD & BLOB embedding storage
├── auth.py              # User registration, bcrypt password login, and session management
├── voice.py             # Voice enrollment, verification routes, and similarity scoring
├── ml/
│   ├── preprocess.py    # Audio loading, resampling, silence removal, normalization
│   ├── features.py      # Multi-channel MFCC, delta, delta-delta & acoustic descriptors
│   └── model.py         # PyTorch VoiceEmbeddingCNN + speaker representation extractor
├── static/
│   ├── recorder.js      # Web Audio API in-browser recorder, PCM WAV encoder & visualizer
│   └── style.css        # Responsive dark UI styling
├── templates/
│   ├── index.html       # Login page (Password & Voice login modes)
│   ├── register.html    # User registration page
│   └── dashboard.html   # User dashboard & voice enrollment / verification testing
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

---

## Features

- **Biometric Voice Enrollment:** Captures microphone audio in-browser, encodes to 16-bit PCM WAV (16kHz mono), cleans and normalizes speech, and generates a compact 128-dimensional embedding vector stored in SQLite.
- **Biometric Voice Verification:** Computes cosine similarity between live test audio and stored enrollment templates with confidence score feedback.
- **Dual Login Modes:** Supports both traditional username + bcrypt password authentication and biometric voice login.
- **Real-Time Visualizer:** Live canvas audio spectrum visualizer in the browser.
- **Offline & Standalone ML Engine:** PyTorch Convolutional Neural Network with Temporal Statistical Pooling (duration-invariant) and multi-channel MFCC features.

---

## Quickstart

### 1. Activate Virtual Environment

**Windows:**
```powershell
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask Server

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`.

---

## ML Pipeline

1. **Preprocessing (`ml/preprocess.py`):**
   - Audio is converted to mono and resampled to 16,000 Hz.
   - Energy-based silence trimming removes non-speech leading/trailing silence.
   - Amplitude normalization balances recording variations.

2. **Feature Extraction (`ml/features.py`):**
   - Extracts 40 Mel-Frequency Cepstral Coefficients (MFCCs).
   - Computes 1st order (delta) and 2nd order (delta-delta) temporal velocity/acceleration features.
   - Computes vocal-tract acoustic statistics (spectral centroid, bandwidth, rolloff, zero-crossing rate).

3. **CNN Embedding (`ml/model.py`):**
   - Deep Convolutional Neural Network processes multi-channel spectral features.
   - Temporal Statistical Pooling aggregates mean and variance across time frames for length invariance.
   - Projects representation into an L2-normalized 128-dimensional biometric embedding.
   - Verification uses cosine similarity metric with matching threshold $\ge 0.82$.
