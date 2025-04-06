import os
import threading
import asyncio
import time
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QSpinBox,
    QGroupBox, QComboBox, QApplication
)
from PyQt5.QtCore import Qt, pyqtSlot, QMetaObject, Q_ARG, QObject, pyqtSignal, QSettings
from PyQt5.QtGui import QKeySequence
import keyboard
from hotkey_input import HotkeyInput
from prompts import get_all_prompts, get_prompt, load_prompts
from translations import TRANSLATIONS
from tts_manager import TTSManager

# Create a signal class for thread-safe communication
class UISignals(QObject):
    analysis_result = pyqtSignal(str)
    status_update = pyqtSignal(str)
    commentary_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    video_analysis_status = pyqtSignal(str)

class ScreenRecorderUI(QMainWindow):
    def __init__(self, recorder):
        super().__init__()
        
        # Store recorder and TTS manager references
        self.recorder = recorder
        
        # Create QSettings for saving/loading preferences
        self.settings = QSettings("ScreenCasting", "ScreenRecorderApp")
        
        # Set default language and load from settings if available
        self.current_language = self.settings.value("language", "en")
        
        # Load prompts for the current language
        load_prompts(self.current_language)
        
        # Create signals for cross-thread communication
        self.signals = UISignals()
        
        # Connect our own signals
        self.signals.analysis_result.connect(self.update_analysis_result)
        self.signals.status_update.connect(self.update_status)
        self.signals.commentary_received.connect(self.update_commentary)
        self.signals.error_occurred.connect(self.update_error_occurred)
        
        # Initialize UI
        self.initUI()
        
        # Load saved settings for controls after UI initialization
        self.load_saved_settings()

        # Create the TTS manager first
        self.tts_manager = TTSManager(self.current_language)
        print("TTS manager created")

        # Connect signals from TTS manager
        self.tts_manager.signals.speech_complete.connect(self.on_speech_complete)
        self.tts_manager.signals.error_occurred.connect(self.on_tts_error)

         # Wait for the TTS model to load before initializing the UI
        self.wait_for_tts_model_loaded()
        self.update_tts_properties()
        self.load_saved_tts_settings()

    def update_tts_properties(self):
        # Get current volume from local TTS manager (0.0-1.0) and convert to percentage (0-100)
        current_volume = int(self.tts_manager.get_tts_properties().get('volume', 1.0) * 100)
        self.volume_slider.setValue(current_volume)
        # Get current rate from TTS manager (default is usually 150 WPM)
        current_rate = self.tts_manager.get_tts_properties().get('rate', 150)
        self.speed_slider.setValue(int(current_rate))
        # Populate with available voices
        available_voices = self.tts_manager.get_available_voices()
        selected_prompt_title = self.prompt_combo.currentText()
        self.tts_manager.set_tts_properties(voice=get_prompt(selected_prompt_title)["default_voice"])
        current_voice_id = self.tts_manager.get_tts_properties().get('voice', None)
        
        # Add the "Default (Use prompt's voice)" option as the first item
        self.voice_combo.addItem(TRANSLATIONS[self.current_language]["use_prompt_voice"], "default")
        
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
        
        # Populate with available audio devices
        available_devices = self.tts_manager.get_audio_devices()
        current_device = self.tts_manager.get_current_audio_device()
        
        # Add the "System Default" option as the first item
        self.audio_device_combo.addItem(TRANSLATIONS[self.current_language]["system_default"], None)
        
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

    def on_tts_error(self, error_msg):
        """Handle errors from the TTS manager"""
        self.signals.video_analysis_status.emit("error")
        print(f"TTS ERROR DETECTED: {error_msg}")
        self.tts_error_detected = True
        self.tts_error_message = error_msg
        
        print(f"TTS ERROR DETECTED: {error_msg}")
        print(hasattr(self, 'loading_label'))
        # Update the loading screen to show the error
        if hasattr(self, 'loading_label'):
            self.loading_label.setText(f"Error loading TTS model: {error_msg}")
            self.splash.setStyleSheet("background-color: #ffeeee; border: 1px solid #ff6666;")
            
            # Add a simple OK button that exits immediately
            layout = self.splash.layout()
            if layout.count() < 3:  # Make sure we don't add it twice
                ok_btn = QPushButton("OK")
                ok_btn.clicked.connect(lambda: os._exit(1))  # Force exit to restart
                layout.addWidget(ok_btn)
            
            # Force UI updates
            for i in range(10):  # Multiple processEvents calls to ensure UI updates
                QApplication.processEvents()
    
    def load_saved_tts_settings(self):
        # TTS Volume
        volume = self.settings.value("tts_volume", 100, type=int)
        self.volume_slider.setValue(volume)
        self.update_tts_volume(volume)
        
        # TTS Speed
        speed = self.settings.value("tts_speed", 150, type=int)
        self.speed_slider.setValue(speed)
        self.update_tts_speed(speed)

        # TTS Voice
        voice_id = self.settings.value("tts_voice", "default")
        if voice_id != "default":
            index = self.voice_combo.findData(voice_id)
            if index >= 0:
                self.voice_combo.setCurrentIndex(index)
                self.update_tts_voice(index)
        
        # Audio Device
        device_id = self.settings.value("audio_device", None)
        if device_id:
            index = self.audio_device_combo.findData(device_id)
            if index >= 0:
                self.audio_device_combo.setCurrentIndex(index)
                self.update_audio_device(index)

    def load_saved_settings(self):
        """Load all saved settings and apply to UI controls"""
        # Buffer size
        buffer_size = self.settings.value("buffer_size", self.recorder.buffer_size, type=int)
        self.buffer_size_spinner.setValue(buffer_size)
        self.update_buffer_size(buffer_size)
        
        # API Key
        api_key = self.settings.value("api_key", "")
        if api_key:
            self.api_key_input.setText(api_key)
            
        # Hotkey
        hotkey = self.settings.value("hotkey", "Pause")
        self.hotkey_input.setText(hotkey)
        
        # Prompt selection
        prompt_title = self.settings.value("selected_prompt", "")
        if prompt_title:
            index = self.prompt_combo.findText(prompt_title)
            if index >= 0:
                self.prompt_combo.setCurrentIndex(index)
    
    def initUI(self):
        self.setWindowTitle(TRANSLATIONS[self.current_language]["window_title"])
        self.setGeometry(100, 100, 600, 450)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Add language selector at the top
        language_layout = QHBoxLayout()
        language_layout.addWidget(QLabel(TRANSLATIONS[self.current_language]["language"]))
        self.language_combo = QComboBox()
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Русский", "ru")
        
        # Set the current language in the combo box
        index = self.language_combo.findData(self.current_language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
            
        self.language_combo.currentIndexChanged.connect(self.change_language)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        main_layout.addLayout(language_layout)
        
        # Status label
        self.status_label = QLabel(TRANSLATIONS[self.current_language]["status_ready"])
        main_layout.addWidget(self.status_label)
        
        # Recording controls section
        recording_section = QGroupBox(TRANSLATIONS[self.current_language]["recording_controls"])
        recording_section.setObjectName("recording_section")
        recording_layout = QHBoxLayout(recording_section)
        
        # Buffer size control
        buffer_layout = QHBoxLayout()
        self.buffer_label = QLabel(TRANSLATIONS[self.current_language]["buffer_size"])
        buffer_layout.addWidget(self.buffer_label)
        self.buffer_size_spinner = QSpinBox()
        self.buffer_size_spinner.setRange(5, 60)  # Allow 5-60 seconds
        self.buffer_size_spinner.setSingleStep(5)
        self.buffer_size_spinner.setSuffix(TRANSLATIONS[self.current_language]["buffer_seconds"])
        self.buffer_size_spinner.setValue(self.recorder.buffer_size)  # Use the current value from recorder
        self.buffer_size_spinner.setToolTip("Number of seconds of footage kept in memory")
        self.buffer_size_spinner.valueChanged.connect(self.update_buffer_size)
        buffer_layout.addWidget(self.buffer_size_spinner)
        recording_layout.addLayout(buffer_layout)
        
        # Start capture button
        self.start_button = QPushButton(TRANSLATIONS[self.current_language]["start_recording"])
        self.start_button.clicked.connect(self.start_capture)
        recording_layout.addWidget(self.start_button)
        
        # Stop capture button
        self.stop_button = QPushButton(TRANSLATIONS[self.current_language]["stop_recording"])
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        recording_layout.addWidget(self.stop_button)
        
        main_layout.addWidget(recording_section)
        
        # Analysis section
        analysis_section = QGroupBox(TRANSLATIONS[self.current_language]["analysis_controls"])
        analysis_section.setObjectName("analysis_section")
        analysis_layout = QVBoxLayout(analysis_section)
        
        # Prompt selection dropdown
        prompt_layout = QHBoxLayout()
        self.prompt_label = QLabel(TRANSLATIONS[self.current_language]["prompt"])
        prompt_layout.addWidget(self.prompt_label)
        self.prompt_combo = QComboBox()

        # Populate dropdown with prompt titles
        for prompt in get_all_prompts():
            self.prompt_combo.addItem(prompt["title"])
        
        # Connect signal to save selected prompt
        self.prompt_combo.currentTextChanged.connect(self.save_selected_prompt)
        
        prompt_layout.addWidget(self.prompt_combo)
        analysis_layout.addLayout(prompt_layout)
        
        # Run analysis button
        analysis_button_layout = QHBoxLayout()
        self.analyze_button = QPushButton(TRANSLATIONS[self.current_language]["run_analysis"])
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.run_analysis)
        analysis_button_layout.addWidget(self.analyze_button)
        
        # Hotkey configuration
        self.hotkey_label = QLabel(TRANSLATIONS[self.current_language]["hotkey"])
        analysis_button_layout.addWidget(self.hotkey_label)
        self.hotkey_input = HotkeyInput()
        self.hotkey_input.setText("Pause")  # Set default hotkey
        # Connect signal to save hotkey
        self.hotkey_input.textChanged.connect(self.save_hotkey)
        analysis_button_layout.addWidget(self.hotkey_input)
        
        analysis_layout.addLayout(analysis_button_layout)
        
        # TTS volume control
        volume_layout = QHBoxLayout()
        self.volume_label = QLabel(TRANSLATIONS[self.current_language]["tts_volume"])
        volume_layout.addWidget(self.volume_label)
        self.volume_slider = QSpinBox()
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setSuffix(TRANSLATIONS[self.current_language]["volume_percent"])
        
        self.volume_slider.setToolTip("Adjust text-to-speech volume")
        self.volume_slider.valueChanged.connect(self.update_tts_volume)
        volume_layout.addWidget(self.volume_slider)
        
        # TTS speed control
        self.speed_label = QLabel(TRANSLATIONS[self.current_language]["tts_speed"])
        volume_layout.addWidget(self.speed_label)
        self.speed_slider = QSpinBox()
        self.speed_slider.setRange(50, 300)
        self.speed_slider.setSingleStep(10)
        self.speed_slider.setSuffix(TRANSLATIONS[self.current_language]["speed_wpm"])
        self.speed_slider.setToolTip("Adjust text-to-speech speed (words per minute)")
        self.speed_slider.valueChanged.connect(self.update_tts_speed)
        volume_layout.addWidget(self.speed_slider)
        
        # Add a new row for voice selection
        voice_layout = QHBoxLayout()
        self.voice_label = QLabel(TRANSLATIONS[self.current_language]["tts_voice"])
        voice_layout.addWidget(self.voice_label)
        self.voice_combo = QComboBox()
        self.voice_combo.setToolTip("Select text-to-speech voice")
        
        # Set minimum width for the combo to ensure names are readable
        self.voice_combo.setMinimumWidth(200)
        
        # Always select the "Default" option by default (index 0)
        self.voice_combo.setCurrentIndex(0)
        self.voice_combo.currentIndexChanged.connect(self.update_tts_voice)
        voice_layout.addWidget(self.voice_combo)
        
        # Add audio device selection
        audio_device_layout = QHBoxLayout()
        self.audio_device_label = QLabel(TRANSLATIONS[self.current_language]["audio_device"])
        audio_device_layout.addWidget(self.audio_device_label)
        self.audio_device_combo = QComboBox()
        self.audio_device_combo.setToolTip("Select audio output device for TTS")
        
        self.audio_device_combo.currentIndexChanged.connect(self.update_audio_device)
        audio_device_layout.addWidget(self.audio_device_combo)
        
        # Add voice layout to main layout
        analysis_layout.addLayout(voice_layout)
        analysis_layout.addLayout(audio_device_layout)
        
        analysis_layout.addLayout(volume_layout)
        main_layout.addWidget(analysis_section)
        
        # API Key section
        api_section = QGroupBox(TRANSLATIONS[self.current_language]["gemini_api"])
        api_section.setObjectName("api_section")
        api_layout = QVBoxLayout(api_section)
        
        # API Key input
        self.api_key_label = QLabel(TRANSLATIONS[self.current_language]["api_key"])
        api_layout.addWidget(self.api_key_label)
        self.api_key_input = QTextEdit()
        self.api_key_input.setMaximumHeight(30)
        
        # Set the API key from environment if available
        if self.recorder.gemini_api.api_key:
            self.api_key_input.setText(self.recorder.gemini_api.api_key)
        else:
            self.api_key_input.setPlaceholderText(TRANSLATIONS[self.current_language]["api_key_placeholder"])
        
        api_layout.addWidget(self.api_key_input)
        main_layout.addWidget(api_section)
        
        # Results section
        results_section = QGroupBox(TRANSLATIONS[self.current_language]["analysis_results"])
        results_section.setObjectName("results_section")
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
        # Save to settings
        self.settings.setValue("buffer_size", value)
        print(f"Buffer size set to {value} seconds")
    
    @pyqtSlot()
    def start_capture(self):
        # provide api key from input field
        api_key = self.api_key_input.toPlainText().strip()
        # Save API key to settings
        self.settings.setValue("api_key", api_key)
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
        # Save to settings
        self.settings.setValue("tts_volume", value)
        print(f"TTS volume set to {volume:.2f} ({value}%)")
    
    @pyqtSlot(int)
    def update_tts_speed(self, value):
        """Update the text-to-speech speed based on slider value"""
        # Convert WPM to rate multiplier (assuming 150 WPM is normal speed or 1.0 rate)
        rate_multiplier = value / 150.0
        self.tts_manager.set_tts_properties(rate=rate_multiplier)
        # Save to settings
        self.settings.setValue("tts_speed", value)
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
            
            # Save to settings
            self.settings.setValue("tts_voice", voice_id)
            
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
        
        # Save to settings
        self.settings.setValue("audio_device", device_id)
        
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
        """Translate and update the status message"""
        # Translate common status messages
        if status == "Ready":
            self.status_label.setText(TRANSLATIONS[self.current_language]["status_ready"])
        elif status == "Recording active...":
            self.status_label.setText(TRANSLATIONS[self.current_language]["status_recording"])
        elif status == "Recording stopped.":
            self.status_label.setText(TRANSLATIONS[self.current_language]["status_stopped"])
        elif status == "Analyzing video...":
            self.status_label.setText(TRANSLATIONS[self.current_language]["status_analyzing"])
        elif status == "Analysis complete.":
            self.status_label.setText(TRANSLATIONS[self.current_language]["status_analysis_complete"])
        elif status.startswith("Error:"):
            error_msg = status[6:]  # Remove "Error:" prefix
            self.status_label.setText(TRANSLATIONS[self.current_language]["status_error"].format(error_msg))
        else:
            # For other messages, just display as-is
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

    @pyqtSlot(int)
    def change_language(self, index):
        """Change the UI language"""
        lang_code = self.language_combo.itemData(index)
        if lang_code != self.current_language:
            self.current_language = lang_code
            # Save the selected language to settings
            self.settings.setValue("language", lang_code)
            
            # Load prompts for the new language
            load_prompts(self.current_language)
            
            # Update prompt combo box
            self.prompt_combo.clear()
            for prompt in get_all_prompts():
                self.prompt_combo.addItem(prompt["title"])
            
            self.update_ui_texts()
            
            # After updating UI, reload saved settings for the new language
            self.load_saved_settings()
    
    def update_ui_texts(self):
        """Update all UI texts based on the current language"""
        # Window title
        self.setWindowTitle(TRANSLATIONS[self.current_language]["window_title"])
        
        # Status label
        self.status_label.setText(TRANSLATIONS[self.current_language]["status_ready"])
        
        # Recording controls
        recording_section = self.findChild(QGroupBox, "recording_section")
        if recording_section:
            recording_section.setTitle(TRANSLATIONS[self.current_language]["recording_controls"])
        
        self.buffer_label.setText(TRANSLATIONS[self.current_language]["buffer_size"])
        self.buffer_size_spinner.setSuffix(TRANSLATIONS[self.current_language]["buffer_seconds"])
        self.start_button.setText(TRANSLATIONS[self.current_language]["start_recording"])
        self.stop_button.setText(TRANSLATIONS[self.current_language]["stop_recording"])
        
        # Analysis controls
        analysis_section = self.findChild(QGroupBox, "analysis_section")
        if analysis_section:
            analysis_section.setTitle(TRANSLATIONS[self.current_language]["analysis_controls"])
        
        self.prompt_label.setText(TRANSLATIONS[self.current_language]["prompt"])
        self.analyze_button.setText(TRANSLATIONS[self.current_language]["run_analysis"])
        self.hotkey_label.setText(TRANSLATIONS[self.current_language]["hotkey"])
        
        self.volume_label.setText(TRANSLATIONS[self.current_language]["tts_volume"])
        self.volume_slider.setSuffix(TRANSLATIONS[self.current_language]["volume_percent"])
        self.speed_label.setText(TRANSLATIONS[self.current_language]["tts_speed"])
        self.speed_slider.setSuffix(TRANSLATIONS[self.current_language]["speed_wpm"])
        self.voice_label.setText(TRANSLATIONS[self.current_language]["tts_voice"])
        
        # Update combo box items
        current_voice_id = self.voice_combo.itemData(0)
        self.voice_combo.setItemText(0, TRANSLATIONS[self.current_language]["use_prompt_voice"])
        
        self.audio_device_label.setText(TRANSLATIONS[self.current_language]["audio_device"])
        self.audio_device_combo.setItemText(0, TRANSLATIONS[self.current_language]["system_default"])
        
        # API section
        api_section = self.findChild(QGroupBox, "api_section")
        if api_section:
            api_section.setTitle(TRANSLATIONS[self.current_language]["gemini_api"])
        
        self.api_key_label.setText(TRANSLATIONS[self.current_language]["api_key"])
        if not self.api_key_input.toPlainText():
            self.api_key_input.setPlaceholderText(TRANSLATIONS[self.current_language]["api_key_placeholder"])
        
        # Results section
        results_section = self.findChild(QGroupBox, "results_section")
        if results_section:
            results_section.setTitle(TRANSLATIONS[self.current_language]["analysis_results"]) 

    @pyqtSlot(str)
    def save_selected_prompt(self, prompt_title):
        """Save the selected prompt to settings"""
        self.settings.setValue("selected_prompt", prompt_title)
        print(f"Saved selected prompt: {prompt_title}")
    
    @pyqtSlot()
    def save_hotkey(self):
        """Save the current hotkey to settings"""
        hotkey = self.hotkey_input.toPlainText()
        self.settings.setValue("hotkey", hotkey)
        print(f"Saved hotkey: {hotkey}")

    @pyqtSlot()
    def run_analysis(self):
        """Handle the analyze button click event"""
        # Save the currently selected prompt
        prompt_title = self.prompt_combo.currentText()
        self.settings.setValue("selected_prompt", prompt_title)
        
        # Set the status
        self.status_label.setText(TRANSLATIONS[self.current_language]["status_analyzing"])
        
        # Get the selected prompt details
        prompt = get_prompt(prompt_title)
        
        # Check if we should use the prompt's default voice
        if self.voice_combo.currentData() == "default" and "default_voice" in prompt:
            # Use the prompt's default voice
            self.tts_manager.set_tts_properties(voice=prompt["default_voice"])
        
        # TODO: Call the actual analysis function in the recorder with the selected prompt
        # This will depend on how your application is structured
        
        print(f"Running analysis with prompt: {prompt_title}") 