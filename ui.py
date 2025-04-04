import os
import threading
import asyncio
import time
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QSpinBox,
    QGroupBox, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSlot, QMetaObject, Q_ARG, QObject, pyqtSignal
from PyQt5.QtGui import QKeySequence
import keyboard
from hotkey_input import HotkeyInput
from prompts import prompts, get_prompt

# Create a signal class for thread-safe communication
class UISignals(QObject):
    analysis_result = pyqtSignal(str)
    status_update = pyqtSignal(str)
    commentary_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    video_analysis_status = pyqtSignal(str)

class ScreenRecorderUI(QMainWindow):
    def __init__(self, recorder, tts_manager):
        super().__init__()
        
        # Store recorder and TTS manager references
        self.recorder = recorder
        self.tts_manager = tts_manager
        
        # Create signals for cross-thread communication
        self.signals = UISignals()
        
        # Connect signals from recorder
        self.signals.commentary_received.connect(self.update_commentary)
        self.signals.error_occurred.connect(self.update_error_occurred)
        
        # Connect signals from TTS manager
        self.tts_manager.signals.speech_complete.connect(self.on_speech_complete)
        
        # Connect our own signals
        self.signals.analysis_result.connect(self.update_analysis_result)
        self.signals.status_update.connect(self.update_status)
        self.signals.commentary_received.connect(self.update_commentary)
        self.signals.error_occurred.connect(self.update_error_occurred)
        
        # Initialize UI
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Screen Recording Commentary")
        self.setGeometry(100, 100, 600, 450)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Status label
        self.status_label = QLabel("Ready. Screen recording tool is active.")
        main_layout.addWidget(self.status_label)
        
        # Recording controls section
        recording_section = QGroupBox("Recording Controls")
        recording_layout = QHBoxLayout(recording_section)
        
        # Buffer size control
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Buffer Size:"))
        self.buffer_size_spinner = QSpinBox()
        self.buffer_size_spinner.setRange(5, 60)  # Allow 5-60 seconds
        self.buffer_size_spinner.setSingleStep(5)
        self.buffer_size_spinner.setSuffix(" sec")
        self.buffer_size_spinner.setValue(self.recorder.buffer_size)  # Use the current value from recorder
        self.buffer_size_spinner.setToolTip("Number of seconds of footage kept in memory")
        self.buffer_size_spinner.valueChanged.connect(self.update_buffer_size)
        buffer_layout.addWidget(self.buffer_size_spinner)
        recording_layout.addLayout(buffer_layout)
        
        # Start capture button
        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.start_capture)
        recording_layout.addWidget(self.start_button)
        
        # Stop capture button
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        recording_layout.addWidget(self.stop_button)
        
        main_layout.addWidget(recording_section)
        
        # Analysis section
        analysis_section = QGroupBox("Analysis Controls")
        analysis_layout = QVBoxLayout(analysis_section)
        
        # Prompt selection dropdown
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        self.prompt_combo = QComboBox()
        
        # Populate dropdown with prompt titles
        for prompt in prompts:
            self.prompt_combo.addItem(prompt["title"])
        
        prompt_layout.addWidget(self.prompt_combo)
        analysis_layout.addLayout(prompt_layout)
        
        # Run analysis button
        analysis_button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Run Analysis (Last 5 Seconds)")
        self.analyze_button.setEnabled(False)
        analysis_button_layout.addWidget(self.analyze_button)
        
        # Hotkey configuration
        analysis_button_layout.addWidget(QLabel("Hotkey:"))
        self.hotkey_input = HotkeyInput()
        self.hotkey_input.setText("Pause")  # Set default hotkey
        analysis_button_layout.addWidget(self.hotkey_input)
        
        analysis_layout.addLayout(analysis_button_layout)
        
        # TTS volume control
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("TTS Volume:"))
        self.volume_slider = QSpinBox()
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setSuffix("%")
        # Get current volume from local TTS manager (0.0-1.0) and convert to percentage (0-100)
        current_volume = int(self.tts_manager.get_tts_properties().get('volume', 1.0) * 100)
        self.volume_slider.setValue(current_volume)
        self.volume_slider.setToolTip("Adjust text-to-speech volume")
        self.volume_slider.valueChanged.connect(self.update_tts_volume)
        volume_layout.addWidget(self.volume_slider)
        
        # TTS speed control
        volume_layout.addWidget(QLabel("TTS Speed:"))
        self.speed_slider = QSpinBox()
        self.speed_slider.setRange(50, 300)
        self.speed_slider.setSingleStep(10)
        self.speed_slider.setSuffix(" WPM")
        # Get current rate from TTS manager (default is usually 150 WPM)
        current_rate = self.tts_manager.get_tts_properties().get('rate', 150)
        self.speed_slider.setValue(int(current_rate))
        self.speed_slider.setToolTip("Adjust text-to-speech speed (words per minute)")
        self.speed_slider.valueChanged.connect(self.update_tts_speed)
        volume_layout.addWidget(self.speed_slider)
        
        # Add a new row for voice selection
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("TTS Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.setToolTip("Select text-to-speech voice")
        
        # Populate with available voices
        available_voices = self.tts_manager.get_available_voices()
        selected_prompt_title = self.prompt_combo.currentText()
        self.tts_manager.set_tts_properties(voice=get_prompt(selected_prompt_title)["default_voice"])
        current_voice_id = self.tts_manager.get_tts_properties().get('voice', None)
        
        # Add the "Default (Use prompt's voice)" option as the first item
        self.voice_combo.addItem("✓ Default (Use prompt's voice)", "default")
        
        for i, voice in enumerate(available_voices):
            # Enhanced display format to include more voice information
            language = voice['languages'][0] if voice['languages'] else "unknown"
            language_display = "Russian" if language == "ru" else "English" if language == "en" else language
            
            # Format: "Voice Name (Language)"
            display_name = f"{voice['name']} ({language_display})"
            
            # Add a marker for the current voice
            if current_voice_id and voice['id'] == current_voice_id:
                display_name = f"{display_name}"
            
            self.voice_combo.addItem(display_name, voice['id'])
        
        # Set minimum width for the combo to ensure names are readable
        self.voice_combo.setMinimumWidth(200)
        
        # Always select the "Default" option by default (index 0)
        self.voice_combo.setCurrentIndex(0)
        self.voice_combo.currentIndexChanged.connect(self.update_tts_voice)
        voice_layout.addWidget(self.voice_combo)
        
        # Add audio device selection
        audio_device_layout = QHBoxLayout()
        audio_device_layout.addWidget(QLabel("Audio Device:"))
        self.audio_device_combo = QComboBox()
        self.audio_device_combo.setToolTip("Select audio output device for TTS")
        
        # Populate with available audio devices
        available_devices = self.tts_manager.get_audio_devices()
        current_device = self.tts_manager.get_current_audio_device()
        
        # Add the "System Default" option as the first item
        self.audio_device_combo.addItem("✓ System Default", None)
        
        for device in available_devices:
            display_name = f"{device['name']} ({device['channels']} channels)"
            
            # Mark default device
            if device.get('default', False):
                display_name = f"{display_name} [System Default]"
                
            # Add marker for current device if one is selected
            if current_device and device['id'] == current_device['id']:
                display_name = f"✓ {display_name}"
                
            self.audio_device_combo.addItem(display_name, device['id'])
        
        # Set minimum width for the combo to ensure names are readable
        self.audio_device_combo.setMinimumWidth(200)
        
        # Set the current item
        self.audio_device_combo.setCurrentIndex(0)  # Default is system default
        self.audio_device_combo.currentIndexChanged.connect(self.update_audio_device)
        audio_device_layout.addWidget(self.audio_device_combo)
        
        # Add voice layout to main layout
        analysis_layout.addLayout(voice_layout)
        analysis_layout.addLayout(audio_device_layout)
        
        analysis_layout.addLayout(volume_layout)
        main_layout.addWidget(analysis_section)
        
        # API Key section
        api_section = QGroupBox("Gemini API")
        api_layout = QVBoxLayout(api_section)
        
        # API Key input
        api_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QTextEdit()
        self.api_key_input.setMaximumHeight(30)
        
        # Set the API key from environment if available
        if self.recorder.gemini_api.api_key:
            self.api_key_input.setText(self.recorder.gemini_api.api_key)
        else:
            self.api_key_input.setPlaceholderText("Enter your Gemini API key here")
        
        api_layout.addWidget(self.api_key_input)
        main_layout.addWidget(api_section)
        
        # Results section
        results_section = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_section)
        
        # Analysis results area
        self.analysis_result = QTextEdit()
        self.analysis_result.setReadOnly(True)
        # Set default welcome text
        self.analysis_result.setText("Welcome to Screen Recording Commentary!\n\nCapture your screen and get AI-powered commentary on your activities.")
        results_layout.addWidget(self.analysis_result)
        
        main_layout.addWidget(results_section)
        
        self.setCentralWidget(central_widget)
    
    @pyqtSlot(int)
    def update_buffer_size(self, value):
        """Update the buffer size in the recorder"""
        self.recorder.set_buffer_size(value)
        print(f"Buffer size set to {value} seconds")
    
    @pyqtSlot()
    def start_capture(self):
        # provide api key from input field
        api_key = self.api_key_input.toPlainText().strip()
        self.recorder.start_capture(api_key)
        self.status_label.setText("Capturing screen...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
    
    @pyqtSlot()
    def stop_capture(self):
        self.recorder.stop_capture()
        self.status_label.setText("Capture stopped. Buffer contains last recordings.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        
        # Stop hotkey listener if active
        if self.recorder.is_listening_for_hotkeys:
            self.recorder.stop_printscreen_listener()

    @pyqtSlot(int)
    def update_tts_volume(self, value):
        """Update the text-to-speech volume based on slider value"""
        # Convert percentage (0-100) to decimal (0.0-1.0)
        volume = value / 100.0
        # Update the TTS engine volume directly on our TTSManager
        self.tts_manager.set_tts_properties(volume=volume)
        print(f"TTS volume set to {volume:.2f} ({value}%)")
    
    @pyqtSlot(int)
    def update_tts_speed(self, value):
        """Update the text-to-speech speed based on slider value"""
        # Convert WPM to rate multiplier (assuming 150 WPM is normal speed or 1.0 rate)
        rate_multiplier = value / 150.0
        self.tts_manager.set_tts_properties(rate=rate_multiplier)
        print(f"TTS speed set to {value} WPM (rate multiplier: {rate_multiplier:.2f})")
    
    @pyqtSlot(int)
    def update_tts_voice(self, index):
        """Update the text-to-speech voice based on combo selection"""
        # Get the voice ID from the combo box user data
        voice_id = self.voice_combo.itemData(index)
        if voice_id:
            # If "default" is selected, don't update the TTS manager's voice yet
            # It will be handled when analyzing with the chosen prompt
            if voice_id != "default":
                self.tts_manager.set_tts_properties(voice=voice_id)
            
            voice_name = self.voice_combo.itemText(index)
            print(f"TTS voice set to {voice_name} (ID: {voice_id})")
            
            # Update the display names to mark the currently selected voice
            for i in range(self.voice_combo.count()):
                item_text = self.voice_combo.itemText(i)
                if i == index:
                    # Add checkmark to selected voice if not already there
                    if not item_text.startswith("✓ "):
                        self.voice_combo.setItemText(i, f"✓ {item_text}")
                else:
                    # Remove checkmark from other voices
                    if item_text.startswith("✓ "):
                        self.voice_combo.setItemText(i, item_text[2:])
    
    @pyqtSlot(int)
    def update_audio_device(self, index):
        """Update the audio output device based on combo selection"""
        # Get the device ID from the combo box user data
        device_id = self.audio_device_combo.itemData(index)
        
        # Set the device in the TTS manager
        self.tts_manager.set_audio_device(device_id)
        
        device_name = self.audio_device_combo.itemText(index)
        print(f"Audio output device set to {device_name} (ID: {device_id})")
        
        # Update the display names to mark the currently selected device
        for i in range(self.audio_device_combo.count()):
            item_text = self.audio_device_combo.itemText(i)
            if i == index:
                # Add checkmark to selected device if not already there
                if not item_text.startswith("✓ "):
                    self.audio_device_combo.setItemText(i, f"✓ {item_text}")
            else:
                # Remove checkmark from other devices
                if item_text.startswith("✓ "):
                    self.audio_device_combo.setItemText(i, item_text[2:])
    
    @pyqtSlot(str)
    def update_analysis_result(self, result):
        """Update the analysis result text in a thread-safe way"""
        if not result:
            print("WARNING: Empty analysis result received")
            result = "No analysis results received. Please try again."
            
        print(f"DEBUG: Updating analysis results with text ({len(result)} chars)")
        self.analysis_result.clear()  # Clear existing text first
        # Set font to ensure proper display of special characters and Cyrillic
        font = self.analysis_result.font()
        font.setPointSize(10)  # Slightly larger font for better readability
        self.analysis_result.setFont(font)
        self.analysis_result.setText(result)
        # Auto-scroll to the top
        self.analysis_result.moveCursor(self.analysis_result.textCursor().Start)
    
    @pyqtSlot(str)
    def update_status(self, status):
        """Update the status label in a thread-safe way"""
        self.status_label.setText(status)
    
    @pyqtSlot(str)
    def update_commentary(self, commentary):
        """Update with received commentary in a thread-safe way"""
        if commentary:
            # Make sure text is properly encoded for display
            self.analysis_result.setText(commentary)
            print(f"Commentary received and displayed ({len(commentary)} chars)")
    
    @pyqtSlot(str)
    def update_error_occurred(self, error_message):
        """Handle error messages in a thread-safe way"""
        self.status_label.setText(f"Error: {error_message}")
        self.analysis_result.setText(f"Error occurred: {error_message}")
        print(f"Error displayed: {error_message}")

    @pyqtSlot(str)
    def on_hotkey_pressed(self, hotkey):
        """Handler for when a hotkey is pressed"""
        print(f"UI detected hotkey press: {hotkey}")
        # Disable the analyze button when a hotkey triggers analysis
        self.analyze_button.setEnabled(False)
        self.status_label.setText(f"Hotkey {hotkey} pressed. Processing...")
    
    @pyqtSlot()
    def on_speech_complete(self):
        """Handle speech completion signal from TTS manager"""
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Ready. Speech complete.")
        self.signals.video_analysis_status.emit("completed")
    
    def closeEvent(self, event):
        """Clean up resources when the application is closed"""
        print("closeEvent called - application closing")
        self.recorder.stop_capture()
            
        event.accept() 