"""
Text-to-Speech Manager using Silero TTS model.
This implementation provides high-quality text-to-speech with multiple voices.
"""

import os
import torch
import sounddevice as sd
import soundfile as sf
import threading
import numpy as np
import time
import queue
import re
from urllib.request import urlretrieve
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal
from functools import lru_cache

# Add a signals class for TTS manager
class TTSManagerSignals(QObject):
    speech_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

class TTSManager:
    def __init__(self):
        """Initialize the Silero text-to-speech manager"""
        # Create signals for events
        self.signals = TTSManagerSignals()
        
        # Add explicit error tracking
        self.last_error = None
        
        # Create models directory if it doesn't exist
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model properties
        self.model = None
        self.sample_rate = 48000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.speakers = None
        self.rate = 1.0  # Speed factor
        self.volume = 1.0  # Volume level (0.0 to 1.0)
        self.current_voice = None
        self._russian_voice_id = None
        
        # Audio device settings
        self.audio_device = None  # Use default device initially
        
        # Cache settings
        self.cache_enabled = True
        self.chunk_size = 100  # Max chars per chunk
        
        # Performance settings
        self.use_half_precision = True  # Use FP16 for faster inference if supported
        self.batch_processing = True    # Process multiple chunks in one go
        self.batch_size = 3             # Max number of chunks to process at once
        
        # Create a queue for speech tasks
        self._speech_queue = queue.Queue()
        self._is_speaking = False
        self._stop_requested = False
        
        # Add a lock to prevent multiple threads from calling speak_text simultaneously
        self.tts_lock = threading.Lock()
        
        # Start the speech processing thread
        self._speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self._speech_thread.start()
        
        # Loading model asynchronously to not block initialization
        threading.Thread(target=self.load_model, daemon=True).start()
        
    def load_model(self):
        """Load the Silero TTS model"""
        try:
            model_path = self.models_dir / "silero_tts.pt"
            model_url = "https://models.silero.ai/models/tts/ru/v3_1_ru.pt"
            
            # First check if the model exists directly in the project folder
            project_model_path = Path("v3_1_ru.pt")
            if project_model_path.exists():
                print(f"Found model in project folder: {project_model_path}")
                model_path = project_model_path
            # Otherwise download or use the model in models directory
            elif not model_path.exists():
                print("Downloading Silero TTS model (this may take a moment)...")
                try:
                    urlretrieve(model_url, model_path)
                except Exception as e:
                    error_msg = f"Failed to download TTS model: {str(e)}"
                    print(error_msg)
                    self.last_error = error_msg
                    self.signals.error_occurred.emit(error_msg)
                    return
            
            # Load the model
            print(f"Loading model from {model_path}")
            try:
                self.model = self._download_and_load_model(model_path, model_url)
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}. Attempting to redownload..."
                print(error_msg)
                try:
                    self.model = self._download_and_load_model(model_path, model_url, force_download=True)
                except Exception as e2:
                    error_msg = f"Failed to load model after redownload attempt: {str(e2)}"
                    print(error_msg)
                    self.last_error = error_msg
                    self.signals.error_occurred.emit(error_msg)
                    return
                
            # Apply half precision for faster inference if GPU is available
            if self.use_half_precision and self.device.type == 'cuda':
                try:
                    self.model = self.model.half()
                    print("Using half precision (FP16) for faster inference")
                except Exception as e:
                    print(f"Failed to use half precision: {str(e)}")
                    self.use_half_precision = False
                            
            # Get available speakers
            self.speakers = self.model.speakers
            
            # Set default voice to first Russian speaker
            russian_speakers = [s for s in self.speakers if s.startswith('ru_')]
            self.current_voice = russian_speakers[0] if russian_speakers else self.speakers[0]
            
            # Store Russian voice ID for automatic language detection
            if russian_speakers:
                self._russian_voice_id = russian_speakers[0]
                print(f"Found Russian voice: {self._russian_voice_id}")
            
            print(f"Silero TTS model loaded successfully. Using voice: {self.current_voice}")
        except Exception as e:
            error_msg = f"Error loading Silero TTS model: {str(e)}"
            print(error_msg)
            self.last_error = error_msg
            self.signals.error_occurred.emit(error_msg)
            self.speakers = []  # Ensure speakers list is empty but defined
            
    def _download_and_load_model(self, model_path, model_url, force_download=False):
        """Download (if needed) and load the model
        
        Args:
            model_path: Path to the model file
            model_url: URL to download the model from
            force_download: Whether to force redownload the model
            
        Returns:
            The loaded model
        """
        # Delete model if force_download is True
        if force_download and model_path.exists():
            os.remove(model_path)
            
        # Download the model if it doesn't exist
        if force_download or not model_path.exists():
            print(f"Downloading model from {model_url}...")
            try:
                urlretrieve(model_url, model_path)
            except Exception as e:
                error_msg = f"Failed to download model: {str(e)}"
                print(error_msg)
                self.last_error = error_msg
                self.signals.error_occurred.emit(error_msg)
                raise
            
        # Load the model
        try:
            model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
            model.to(self.device)
            return model
        except Exception as e:
            error_msg = f"Failed to load model file: {str(e)}"
            print(error_msg)
            self.last_error = error_msg
            self.signals.error_occurred.emit(error_msg)
            raise
    
    def _process_speech_queue(self):
        """Process speech tasks from the queue"""
        while True:
            try:
                # Get the next task from the queue, blocking until one is available
                if self._stop_requested:
                    break
                
                # Check if we're already speaking
                if self._is_speaking:
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                    continue
                
                # Get a task from the queue
                text, rate, volume, voice, is_save, filename = self._speech_queue.get(block=True)
                self._is_speaking = True
                
                # Set default values if not provided
                speech_rate = rate if rate is not None else self.rate
                speech_volume = volume if volume is not None else self.volume
                selected_voice = voice if voice is not None else self.current_voice
                
                # Ensure we have a model
                if not self.model:
                    print("TTS model not loaded, skipping speech")
                    self._is_speaking = False
                    self._speech_queue.task_done()
                    time.sleep(1)  # Delay to prevent tight loop
                    continue
                    
                # Ensure we have a valid voice
                if not selected_voice and self.speakers:
                    selected_voice = self.speakers[0]
                    print(f"No voice specified, using {selected_voice}")
                
                # If we have a Russian text, use Russian voice
                has_cyrillic = any(ord('а') <= ord(c) <= ord('я') or ord('А') <= ord(c) <= ord('Я') for c in text)
                if has_cyrillic and self._russian_voice_id and not (selected_voice and selected_voice.startswith('ru_')):
                    selected_voice = self._russian_voice_id
                    print(f"Detected Russian text, using voice: {selected_voice}")
                
                # Acquire lock
                with self.tts_lock:
                    try:
                        # Process text (clean up ellipsis, etc.)
                        processed_text = text.replace('...', ', ')
                        
                        # Generate audio for the entire text - no chunking
                        audio = self._generate_audio(processed_text, selected_voice, speech_rate, speech_volume)
                        if audio is None:
                            raise ValueError("Failed to generate audio")
                        
                        if is_save and filename:
                            # Save to file
                            sf.write(filename, audio.numpy(), self.sample_rate)
                            print(f"Speech saved to file: {filename}")
                        else:
                            # Play the audio with selected device
                            sd.play(audio.numpy(), self.sample_rate, device=self.audio_device)
                            sd.wait()  # Wait until audio is finished playing
                            print("Finished speaking text")
                            
                    except Exception as e:
                        print(f"Error processing speech: {str(e)}")
                        self.signals.error_occurred.emit(str(e))
                
                # Reset flag and mark task as done
                self._is_speaking = False
                self._speech_queue.task_done()
                
                # Emit speech complete signal
                self.signals.speech_complete.emit()
                
                # Add a small delay to ensure the engine has time to reset
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in speech thread: {str(e)}")
                self.signals.error_occurred.emit(str(e))
                time.sleep(0.5)  # Delay to prevent tight loop if there's an error
    
    @lru_cache(maxsize=50)
    def _cached_tts(self, text, speaker):
        """Cached version of TTS generation to avoid regenerating common phrases"""
        return self.model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=self.sample_rate
        )
                
    def _generate_audio(self, text, voice, rate=1.0, volume=1.0):
        """Generate audio for text with the specified parameters"""
        try:
            # Use cached TTS if enabled and parameters allow it
            if self.cache_enabled and rate == 1.0 and volume == 1.0:
                audio = self._cached_tts(text, voice)
            else:
                # Generate audio from text
                audio = self.model.apply_tts(
                    text=text,
                    speaker=voice,
                    sample_rate=self.sample_rate
                )
            
            # Apply rate adjustment if needed (using resampling)
            if rate != 1.0:
                audio = self._adjust_speech_rate(audio, rate)
            
            # Apply volume adjustment
            if volume != 1.0:
                audio = audio * volume
                
            return audio
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return None
    
    def _split_into_sentences(self, text):
        """Split text into sentences for better chunking"""
        # Simple regex to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_batches(self, sentences, batch_size, chunk_size):
        """Create batches of sentences for batch processing"""
        batches = []
        current_batch = []
        current_batch_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed chunk size, start a new chunk
            if current_batch_length + sentence_len > chunk_size and current_batch:
                # If batch has at least one sentence, add it to batches
                if current_batch:
                    batches.append(current_batch)
                current_batch = [sentence]
                current_batch_length = sentence_len
            else:
                # Add sentence to current batch
                current_batch.append(sentence)
                current_batch_length += sentence_len
            
            # If we reached batch_size, add the batch to batches and start a new one
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
                current_batch_length = 0
        
        # Add any remaining sentences in the last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _generate_audio_batch(self, sentences, voice, rate=1.0, volume=1.0):
        """Generate audio for a batch of sentences"""
        generated_chunks = []
        
        # Join sentences with space
        for sentence in sentences:
            chunk_audio = self._generate_audio(sentence, voice, rate, volume)
            if chunk_audio is not None:
                generated_chunks.append(chunk_audio)
        
        return generated_chunks
    
    def _adjust_speech_rate(self, audio, rate):
        """Adjust the speech rate using resampling"""
        if rate == 1.0:
            return audio
            
        # Convert to numpy for processing
        audio_np = audio.numpy()
        
        # Calculate new length based on rate
        new_length = int(len(audio_np) / rate)
        
        # Use numpy to resample the audio
        indices = np.linspace(0, len(audio_np) - 1, new_length)
        indices = indices.astype(np.int32)
        
        # Create new audio with adjusted rate
        new_audio = audio_np[indices]
        
        # Convert back to torch tensor
        return torch.tensor(new_audio)
    
    def speak_text(self, text, rate=None, volume=None, voice=None):
        """Convert text to speech
        
        Args:
            text: Text to be spoken
            rate: Speed of speech (float multiplier, default is 1.0)
            volume: Volume level (0.0 to 1.0)
            voice: Voice ID to use
            
        Returns:
            None
        """
        if not text:
            return
            
        # Add task to the queue
        self._speech_queue.put((text, rate, volume, voice, False, None))
    
    def set_tts_properties(self, rate=None, volume=None, voice=None):
        """
        Set default properties for text-to-speech
        
        Args:
            rate: Speed of speech (float multiplier)
            volume: Volume level (0.0 to 1.0)
            voice: Voice ID to use for speech
            
        Returns:
            Dictionary of current TTS properties
        """
        if rate is not None:
            self.rate = rate
        if volume is not None:
            self.volume = volume
        if voice is not None and self.speakers and voice in self.speakers:
            self.current_voice = voice
        
        # Return current properties
        return self.get_tts_properties()
    
    def get_tts_properties(self):
        """
        Get current text-to-speech properties
        
        Returns:
            Dictionary of current TTS properties
        """
        # Initialize empty speakers list if model failed to load
        if self.speakers is None:
            self.speakers = []
            
        return {
            'rate': self.rate,
            'volume': self.volume,
            'voice': self.current_voice,
            'audio_device': self.audio_device
        }
    
    def set_russian_voice(self):
        """
        Set the text-to-speech engine to use a Russian voice if available
        
        Returns:
            True if successful, False if no Russian voice found
        """
        if self._russian_voice_id:
            self.current_voice = self._russian_voice_id
            return True
        elif self.speakers:
            russian_speakers = [s for s in self.speakers if s.startswith('ru_')]
            if russian_speakers:
                self.current_voice = russian_speakers[0]
                self._russian_voice_id = russian_speakers[0]
                return True
        
        print("No Russian voice found. Using default voice.")
        return False
    
    def get_available_voices(self):
        """
        Get a list of all available TTS voices
        
        Returns:
            List of dictionaries with voice information
        """
        # If the model isn't loaded or failed to load, return empty list or a placeholder voice
        if not self.speakers:
            # Return at least a mock voice so the UI has something to display
            return [{
                'id': 'default',
                'name': 'Default Voice',
                'languages': ['en'],
                'gender': 'unknown',
                'age': 'unknown'
            }]
            
        voice_list = []
        for speaker in self.speakers:
            # Extract language code from speaker ID (typically format: "ru_0", "en_0", etc.)
            parts = speaker.split('_')
            language = parts[0] if len(parts) > 0 else "unknown"
            
            voice_info = {
                'id': speaker,
                'name': speaker,
                'languages': [language],
                'gender': 'unknown',  # Silero doesn't provide gender info
                'age': 'unknown'      # Silero doesn't provide age info
            }
            voice_list.append(voice_info)
            
        return voice_list
        
    def save_to_file(self, text, filename):
        """
        Save speech to an audio file
        
        Args:
            text: Text to convert to speech
            filename: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        if not text or not filename:
            return False
            
        # Add task to the queue
        self._speech_queue.put((text, None, None, None, True, filename))
        return True  # Return immediately (async operation)
    
    def set_cache_enabled(self, enabled=True):
        """
        Enable or disable TTS caching
        
        Args:
            enabled: Whether caching should be enabled
            
        Returns:
            Current cache state
        """
        self.cache_enabled = enabled
        return self.cache_enabled
    
    def set_performance_options(self, use_half_precision=None, batch_processing=None, batch_size=None, chunk_size=None):
        """
        Configure performance options for TTS
        
        Args:
            use_half_precision: Whether to use FP16 for inference
            batch_processing: Whether to process chunks in batches
            batch_size: Maximum number of chunks to process at once
            chunk_size: Maximum size of text chunks
            
        Returns:
            Dictionary of current performance settings
        """
        if use_half_precision is not None:
            # Can only enable half precision if device is cuda and model is loaded
            if use_half_precision and self.device.type == 'cuda' and self.model is not None:
                try:
                    self.model = self.model.half()
                    self.use_half_precision = True
                except:
                    print("Failed to set half precision")
            elif not use_half_precision:
                self.use_half_precision = False
        
        if batch_processing is not None:
            self.batch_processing = batch_processing
        
        if batch_size is not None and batch_size > 0:
            self.batch_size = batch_size
        
        if chunk_size is not None and chunk_size > 0:
            self.chunk_size = chunk_size
        
        return {
            'use_half_precision': self.use_half_precision,
            'batch_processing': self.batch_processing,
            'batch_size': self.batch_size,
            'chunk_size': self.chunk_size
        }
    
    def clear_cache(self):
        """Clear the TTS cache"""
        if hasattr(self._cached_tts, 'cache_clear'):
            self._cached_tts.cache_clear()
            return True
        return False
    
    def get_audio_devices(self):
        """
        Get a list of available audio output devices
        
        Returns:
            List of dictionaries with device information
        """
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_output_channels'] > 0:
                    devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_output_channels'],
                        'default': device.get('default_output', False)
                    })
        except Exception as e:
            print(f"Error getting audio devices: {str(e)}")
        
        return devices
    
    def set_audio_device(self, device_id=None):
        """
        Set the audio output device
        
        Args:
            device_id: Device ID (int) or None to use default
            
        Returns:
            Current device ID or None if using default
        """
        try:
            if device_id is not None:
                # Validate device ID
                devices = sd.query_devices()
                if 0 <= device_id < len(devices) and devices[device_id]['max_output_channels'] > 0:
                    self.audio_device = device_id
                    print(f"Audio device set to: {devices[device_id]['name']}")
                else:
                    print(f"Invalid device ID: {device_id}")
                    self.audio_device = None
            else:
                # Reset to default device
                self.audio_device = None
                print("Using default audio device")
                
        except Exception as e:
            print(f"Error setting audio device: {str(e)}")
            self.audio_device = None
            
        return self.audio_device
    
    def get_current_audio_device(self):
        """
        Get information about the currently selected audio device
        
        Returns:
            Dictionary with device information or None if using default
        """
        if self.audio_device is None:
            return None
            
        try:
            devices = sd.query_devices()
            if 0 <= self.audio_device < len(devices):
                device = devices[self.audio_device]
                return {
                    'id': self.audio_device,
                    'name': device['name'],
                    'channels': device['max_output_channels']
                }
        except Exception as e:
            print(f"Error getting current audio device: {str(e)}")
            
        return None 