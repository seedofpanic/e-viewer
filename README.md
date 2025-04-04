# NVIDIA Screen Recorder

A desktop application that continuously records your screen and provides AI-powered analysis using Google's Gemini API.

## Features

- Continuous background recording with minimal performance impact
- Maintains a configurable buffer of your screen
- Hardware acceleration using NVIDIA technologies when available
- AI-powered analysis of your screen recordings using Google Gemini
- Text-to-speech functionality to read AI commentaries
- Configurable hotkeys and volume control

## Requirements

- Python 3.6+
- NVIDIA GPU (optional, for hardware acceleration)
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
4. AI commentary will appear in the results area and be read aloud
5. Adjust volume as needed using the slider
6. Click "Stop Recording" when finished

## Getting a Google Gemini API Key

Visit the [Google AI Studio](https://makersuite.google.com/app/apikey), sign in, and create an API key. 