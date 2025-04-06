# E-Viewer Screen Recorder

A desktop application that continuously records your screen and provides AI-powered analysis using Google's Gemini API.

## Overview

E-Viewer was primarily designed to create the experience of having a friend watch you play video games. The AI analyzes your gameplay in real-time, providing commentary, suggestions, and reactions as if a friend was sitting beside you during your gaming sessions.

## Features

- Continuous background recording with configurable buffer
- Real-time screen capture with minimal performance impact
- AI-powered analysis of your screen recordings using Google Gemini API
- Text-to-speech functionality to read AI commentaries
- Configurable hotkeys and volume control
- Multiple language support

## Requirements

- Python 3.6+
- NVIDIA GPU (recommended for hardware acceleration)
- Google Gemini API key

## Installation

1. Clone this repository
2. Set up a virtual environment (recommended)
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install the required packages
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory with your API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

1. Run the application: `python main.py`
2. Click "Start Recording" to begin capturing your screen
3. Use the "Run Analysis" button or configured hotkey to analyze your screen
4. AI commentary will appear in the results area and be read aloud (if TTS is enabled)
5. Adjust volume as needed using the slider
6. Click "Stop Recording" when finished

## Getting a Google Gemini API Key

Visit the [Google AI Studio](https://makersuite.google.com/app/apikey), sign in, and create an API key.

## Contact

Author: seedofpanic41@gmail.com 