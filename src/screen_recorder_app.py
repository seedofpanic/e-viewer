import os
import threading
import asyncio
import time
import cv2
import tempfile

import keyboard
from recorder import ScreenRecorder
from tts_manager import TTSManager
from ui import ScreenRecorderUI
from prompts import get_prompt, get_random_prompt
from PyQt5.QtWidgets import QSplashScreen, QLabel, QVBoxLayout, QWidget, QProgressBar, QApplication, QDialog, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont


class ScreenRecorderApp(ScreenRecorderUI):
    def __init__(self):
        print("Initializing ScreenRecorderApp")

        # Initialize recorder and pass TTS manager to it
        self.recorder = ScreenRecorder()
        print("Recorder initialized")

        # Create a loading screen
        self.create_loading_screen("Loading TTS Model...")

        # Set up error detection
        self.tts_error_detected = False
        self.tts_error_message = ""

        # Initialize the UI with our recorder and TTS manager
        print("Calling UI initialization")
        super().__init__(recorder=self.recorder)
        print("UI initialization complete")

        # Will be set by main.py
        print("ScreenRecorderApp initialization complete")

        # Flag to prevent multiple analysis from running simultaneously
        self.analyzing_video = False

        # Close the loading screen if it's still visible
        if hasattr(self, 'splash') and self.splash.isVisible():
            self.splash.close()

        self.analysis_in_progress = False
        self.signals.video_analysis_status.connect(
            self.update_video_analysis_status)
        self.hotkey_input.hotkeyChanged.connect(self.on_hotkey_changed)
        self.analyze_button.clicked.connect(self._trigger_video_analysis)
        self.hotkey_input.init()

    def on_hotkey_changed(self, hotkey):
        # listen for hotkey
        print(f"Listening for hotkey: {hotkey}")
        keyboard.add_hotkey(hotkey, self._trigger_video_analysis)

    def update_video_analysis_status(self, status):
        if status == "error" or status == "completed":
            self.analysis_in_progress = False
            self.signals.status_update.emit("Ready")

    def create_loading_screen(self, message):
        """Create a loading screen with message"""
        self.splash = QDialog(
            None, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.splash.resize(400, 150)
        self.splash.setStyleSheet(
            "background-color: white; border: 1px solid #cccccc;")

        # Create layout
        layout = QVBoxLayout(self.splash)
        layout.setContentsMargins(20, 20, 20, 20)

        # Add message label
        self.loading_label = QLabel(message)
        font = QFont()
        font.setPointSize(12)
        self.loading_label.setFont(font)
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)

        self.splash.show()

        # Process events to ensure the splash screen is displayed
        QApplication.processEvents()

    def wait_for_tts_model_loaded(self):
        """Wait for the TTS model to load"""
        start_time = time.time()
        self.loading_label.setText("Loading TTS Model...")

        # Poll for model load status or error
        while self.tts_manager.model is None:
            # Process events to keep UI responsive
            QApplication.processEvents()

            # Check if an error was detected
            if self.tts_error_detected:
                print("Error detected in the wait loop, waiting for user input")
                # Error handling is already done in on_tts_error
                # Wait for user to click OK (this keeps the app running until user interaction)
                while True:
                    QApplication.processEvents()
                    time.sleep(0.1)
                return

            # Add a direct check for console errors
            # Check stderr for errors
            import sys
            if hasattr(sys.stderr, "getvalue"):
                stderr_value = sys.stderr.getvalue()
                if "error" in stderr_value.lower() or "exception" in stderr_value.lower():
                    self.on_tts_error(
                        f"Error detected in console: {stderr_value[-200:]}")

            time.sleep(0.1)

            # Update loading message with elapsed time
            elapsed = time.time() - start_time
            self.loading_label.setText(
                f"Loading TTS Model... ({elapsed:.1f}s)")

            # If loading takes too long (over 30 seconds), break to prevent hanging
            if elapsed > 30:
                self.loading_label.setText(
                    "TTS Model load timeout. Please restart the application.")
                self.splash.setStyleSheet(
                    "background-color: #ffeeee; border: 1px solid #ff6666;")

                # Add a simple OK button that exits immediately
                ok_btn = QPushButton("OK")
                # Force exit to restart
                ok_btn.clicked.connect(lambda: os._exit(1))

                # Add it to the splash layout
                layout = self.splash.layout()
                if layout.count() < 3:  # Make sure we don't add it twice
                    layout.addWidget(ok_btn)

                QApplication.processEvents()

                # Wait for user to click OK (this keeps the app running until user interaction)
                while True:
                    QApplication.processEvents()
                    time.sleep(0.1)
                return

        # Show success message before closing
        self.loading_label.setText("TTS Model loaded successfully!")
        QApplication.processEvents()
        time.sleep(0.5)

    async def get_sarcastic_commentary(self, api_key=None, context_id="sarcastic_friend", tts_enabled=True, tts_volume=None, tts_rate=None, tts_voice=None, prompt_title=None):
        """
        Get commentary from Gemini on the last 5 seconds of video

        Args:
            api_key: Optional API key for Gemini
            context_id: Identifier to store and retrieve context for this conversation
            tts_enabled: Whether to speak the response using text-to-speech
            tts_volume: Volume level (0.0 to 1.0) for text-to-speech
            tts_rate: Speech rate in words per minute for text-to-speech
            tts_voice: Voice ID to use for text-to-speech
            prompt_title: Title of the prompt to use (if None, a random prompt will be used)

        Returns:
            Dictionary containing commentary and timing information
        """
        try:
            # Start timing video preparation
            video_prep_start = time.time()

            # Set the API key for Gemini API client if provided
            if api_key:
                self.recorder.gemini_api.api_key = api_key

            # Step 1: Prepare video frames
            frames = await self.recorder._prepare_video_frames(max_age_seconds=5.0)
            if not frames:
                return {"commentary": "No frames in the last 5 seconds to analyze", "error": True}

            # Step 2: Encode frames to video
            video_bytes = self.recorder._encode_frames_to_video(frames)
            if not video_bytes:
                return {"commentary": "Failed to encode video", "error": True}

            # Calculate video preparation time
            video_prep_time = time.time() - video_prep_start
            print(f"Video preparation took {video_prep_time:.2f} seconds")

            # Step 3: Get response from Gemini
            api_result = await self._get_gemini_response(video_bytes, context_id, prompt_title)

            # Step 4: Handle TTS output if enabled
            self._handle_tts_output(
                api_result["commentary"],
                tts_enabled,
                tts_volume,
                tts_rate,
                tts_voice
            )

            # Combine results and timing information
            return {
                **api_result,
                "video_prep_time": video_prep_time,
                "total_time": video_prep_time + api_result["api_request_time"]
            }
        except Exception as e:
            error_message = f"Error getting commentary: {str(e)}"
            print(error_message)
            return {"commentary": error_message, "error": True}

    async def _get_gemini_response(self, video_bytes, context_id, prompt_title=None):
        """
        Get commentary from Gemini API

        Args:
            video_bytes: Encoded video bytes
            context_id: Identifier for conversation context
            prompt_title: Specific prompt title to use

        Returns:
            Dictionary with commentary and result information
        """
        # Get the selected prompt or a random one if none is specified
        if prompt_title:
            selected_prompt = get_prompt(prompt_title)
            if not selected_prompt:
                print(
                    f"Prompt with title '{prompt_title}' not found, using random prompt")
                selected_prompt = get_random_prompt()
        else:
            selected_prompt = get_random_prompt()

        print(f"Using prompt: {selected_prompt['title']}")

        # Get previous context if available
        previous_context = self.recorder.context_storage.get(context_id, None)

        # Start timing API request
        api_request_start = time.time()

        # Send to Gemini API with context
        result = await self.recorder.gemini_api.analyze_video(video_bytes, selected_prompt, previous_context)

        # Calculate API request time
        api_request_time = time.time() - api_request_start
        print(f"Gemini API request took {api_request_time:.2f} seconds")

        # Extract just the commentary (removing any explanations Gemini might add)
        commentary = result.strip()

        # Save context for future use
        self.recorder.context_storage[context_id] = {
            "last_response": commentary,
            "timestamp": time.time(),
            "context_saved": True,
        }

        return {
            "commentary": commentary,
            "full_response": result,
            "context_saved": True,
            "api_request_time": api_request_time
        }

    def _handle_tts_output(self, commentary, tts_enabled, tts_volume=None, tts_rate=None, tts_voice=None):
        """
        Handle text-to-speech output for commentary

        Args:
            commentary: Text to speak
            tts_enabled: Whether TTS is enabled
            tts_volume: Volume level for TTS
            tts_rate: Speech rate for TTS  
            tts_voice: Voice to use for TTS

        Returns:
            None
        """
        if not (tts_enabled and commentary and self.tts_manager):
            return

        # Set volume and rate if provided
        tts_props = {}
        if tts_volume is not None:
            tts_props['volume'] = tts_volume
        if tts_rate is not None:
            tts_props['rate'] = tts_rate
        if tts_voice is not None:
            tts_props['voice'] = tts_voice

        # Apply TTS properties if any were set
        if tts_props:
            self.tts_manager.set_tts_properties(**tts_props)

        # Speak the result in a thread-safe way using the recorder's thread pool
        self.recorder.thread_pool.submit(
            self.tts_manager.speak_text, commentary)

    def _trigger_video_analysis(self):
        if self.analysis_in_progress:
            return
        self.analysis_in_progress = True
        self.signals.status_update.emit("Thinking...")

        """Run hotkey-triggered video analysis"""
        # Lock/flag to make sure we don't run multiple analyses at once
        if self.analyzing_video:
            print("Analysis already in progress. Aborting.")
            return

        # Set the flag and update UI if available
        self.analyzing_video = True
        api_key = self.recorder.gemini_api.api_key

        print(f"Hotkey triggered analysis using API key: {api_key[:4]}...")

        # Use async code in a separate thread
        def async_runner():
            try:
                # Create and run an event loop for the background thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Get current TTS properties
                tts_props = self.tts_manager.get_tts_properties()
                tts_volume = tts_props.get('volume', 1.0)
                tts_rate = tts_props.get('rate', 150)
                tts_voice = tts_props.get('voice', None)
                prompt_title = self.prompt_combo.currentText()

                # Run the commentary
                result = loop.run_until_complete(
                    self.get_sarcastic_commentary(api_key, tts_enabled=True, tts_volume=tts_volume,
                                                  tts_rate=tts_rate, tts_voice=tts_voice, prompt_title=prompt_title)
                )

                # Get just the commentary text
                commentary = result.get(
                    "commentary", "Failed to get commentary from Gemini")

                if result.get("error", False):
                    self.signals.video_analysis_status.emit("error")

                # Show message
                print(
                    f"Hotkey-triggered analysis completed in {result.get('total_time', 0):.2f} seconds")

                # Make sure to emit the result to any connected UI
                self.signals.commentary_received.emit(commentary)
            except Exception as e:
                print(f"Error in hotkey-triggered analysis: {str(e)}")
                self.ui.signals.error_occurred.emit(
                    f"Analysis error: {str(e)}")
            finally:
                # Clear flag regardless of success/failure
                self.analyzing_video = False

        # Run the async code in a background thread
        threading.Thread(target=async_runner, daemon=True).start()
