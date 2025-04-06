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
    model_loaded = pyqtSignal(str)  # Signal when a model is loaded with language code

class TTSManager:
    def __init__(self, language_code=None):
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
        self._english_voice_id = None
        self._current_language = language_code  # Default language
        
        # Model URLs by language
        self.model_urls = {
            "ru": "https://models.silero.ai/models/tts/ru/v3_1_ru.pt",
            "en": "https://models.silero.ai/models/tts/en/v3_en.pt",
            "de": "https://models.silero.ai/models/tts/de/v3_de.pt",
            "es": "https://models.silero.ai/models/tts/es/v3_es.pt",
            "fr": "https://models.silero.ai/models/tts/fr/v3_fr.pt"
        }
        
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
        
    def load_model(self, language_code=None):
        """Load the Silero TTS model for specified language or current language
        
        Args:
            language_code: Language code (e.g., 'ru', 'en') or None to use current language
        """
        try:
            # Use specified language or current language
            lang = language_code or self._current_language
            
            # Skip if language is not supported
            if lang not in self.model_urls:
                error_msg = f"Language {lang} is not supported"
                print(error_msg)
                self.last_error = error_msg
                self.signals.error_occurred.emit(error_msg)
                return
            
            model_url = self.model_urls[lang]
            model_filename = f"silero_tts_{lang}.pt"
            model_path = self.models_dir / model_filename
            
            # First check if the model exists directly in the project folder
            project_model_path = Path(f"v3_{lang}.pt" if lang != "en" else "v3_en.pt")
            if project_model_path.exists():
                print(f"Found model in project folder: {project_model_path}")
                model_path = project_model_path
            # Otherwise download or use the model in models directory
            elif not model_path.exists():
                print(f"Downloading Silero TTS model for {lang} (this may take a moment)...")
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
            
            if self.device.type == 'cuda':
                print("Using GPU for inference")
            else:
                print("Using CPU for inference")

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
            
            # Select appropriate default voice
            if lang == "ru":
                # Set default voice to first Russian speaker
                russian_speakers = [s for s in self.speakers if s.startswith('ru_')]
                self.current_voice = russian_speakers[0] if russian_speakers else self.speakers[0]
                # Store Russian voice ID
                if russian_speakers:
                    self._russian_voice_id = russian_speakers[0]
                    print(f"Found Russian voice: {self._russian_voice_id}")
            elif lang == "en":
                # Set default voice to first English speaker
                english_speakers = [s for s in self.speakers if s.startswith('en_')] 
                self.current_voice = english_speakers[0] if english_speakers else self.speakers[0]
                # Store English voice ID
                if english_speakers:
                    self._english_voice_id = english_speakers[0]
                    print(f"Found English voice: {self._english_voice_id}")
            else:
                # For other languages, select first voice that starts with language code
                lang_speakers = [s for s in self.speakers if s.startswith(f'{lang}_')]
                self.current_voice = lang_speakers[0] if lang_speakers else self.speakers[0]
                
            # Update current language
            self._current_language = lang
            print(f"Silero TTS model loaded successfully for language {lang}. Using voice: {self.current_voice}")
            
            # Emit signal that model is loaded
            self.signals.model_loaded.emit(lang)
            
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
                task = self._speech_queue.get(block=True)
                self._is_speaking = True
                
                # Process the speech task
                self._process_speech_task(*task)
                
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
    
    def _process_speech_task(self, text, rate, volume, voice, is_save, filename):
        """Process a single speech task
        
        Args:
            text: Text to be spoken
            rate: Speed of speech (float multiplier)
            volume: Volume level (0.0 to 1.0)
            voice: Voice ID to use
            is_save: Whether to save to file
            filename: Filename to save to (if is_save is True)
        """
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
            return
            
        # Ensure we have a valid voice
        if not selected_voice and self.speakers:
            selected_voice = self.speakers[0]
            print(f"No voice specified, using {selected_voice}")
        
        # Detect language from text and load appropriate model if needed
        detected_lang = self._detect_language(text)
        if detected_lang and detected_lang != self._current_language:
            print(f"Detected language: {detected_lang}, current model language: {self._current_language}")
            print("Loading appropriate language model...")
            
            # Switch model in a new thread to avoid blocking
            loading_thread = threading.Thread(target=self.load_model, args=(detected_lang,), daemon=True)
            loading_thread.start()
            loading_thread.join()  # Wait for model loading to complete
            
            # Automatically select appropriate voice
            if detected_lang == "ru" and self._russian_voice_id:
                selected_voice = self._russian_voice_id
            elif detected_lang == "en" and self._english_voice_id:
                selected_voice = self._english_voice_id
            else:
                # Try to find a voice for the detected language
                lang_speakers = [s for s in self.speakers if s.startswith(f'{detected_lang}_')]
                if lang_speakers:
                    selected_voice = lang_speakers[0]
        
        # Acquire lock
        with self.tts_lock:
            try:
                self.play_test(text, selected_voice, speech_rate, speech_volume, is_save, filename)
                    
            except Exception as e:
                print(f"Error processing speech: {str(e)}")
                self.signals.error_occurred.emit(str(e))

    def _detect_language(self, text):
        """Detect language from text
        
        Args:
            text: Text to detect language from
            
        Returns:
            Language code (e.g., 'ru', 'en') or None if detection fails
        """
        if not text:
            return None
            
        # Check for Cyrillic characters (Russian)
        has_cyrillic = any(ord('а') <= ord(c) <= ord('я') or ord('А') <= ord(c) <= ord('Я') for c in text)
        if has_cyrillic:
            return "ru"
            
        # Simple language detection for other languages
        # This is a very basic implementation - could be improved with proper language detection libraries
        
        # English is default if no other language is detected
        # In a real implementation, you might want to use a proper language detection library
        return "en"
    
    def play_test(self, text, selected_voice, speech_rate, speech_volume, is_save, filename):
        # Process text (clean up ellipsis, etc.)
        processed_text = text.replace('...', ', ')
        
        if is_save and filename:
            # For saving to file, we need the complete audio in one piece
            audio = self._generate_audio(processed_text, selected_voice, speech_rate, speech_volume)
            if audio is None:
                raise ValueError("Failed to generate audio")
            sf.write(filename, audio.numpy(), self.sample_rate)
            print(f"Speech saved to file: {filename}")
        else:
            # Split text into sentences for parallel processing
            sentences = self._split_into_sentences(processed_text)
            batches = self._create_batches(sentences, self.batch_size, self.chunk_size)
            
            # Create a queue for processed audio chunks
            audio_queue = queue.Queue()
            
            # Function to process a batch and add to queue
            def process_batch(batch):
                batch_text = " ".join(batch)
                chunk_audio = self._generate_audio(batch_text, selected_voice, speech_rate, speech_volume)
                if chunk_audio is not None:
                    audio_queue.put(chunk_audio)
            
            # Start processing threads for each batch
            threads = []
            for batch in batches:
                thread = threading.Thread(target=process_batch, args=(batch,))
                thread.start()
                threads.append(thread)
            
            # Play audio chunks as they become available
            chunks_played = 0
            total_chunks = len(batches)
            
            while chunks_played < total_chunks:
                try:
                    # Get next chunk with timeout
                    audio_chunk = audio_queue.get(timeout=2)
                    
                    # Play the chunk
                    sd.play(audio_chunk.numpy(), self.sample_rate, device=self.audio_device)
                    sd.wait()  # Wait until this chunk finishes playing
                    
                    chunks_played += 1
                    audio_queue.task_done()
                    
                except queue.Empty:
                    # No chunks ready yet, check if processing is still ongoing
                    if all(not t.is_alive() for t in threads):
                        # All threads finished but queue is empty, some chunks failed
                        if chunks_played < total_chunks:
                            print(f"Warning: Only {chunks_played}/{total_chunks} chunks were processed successfully")
                        break
            
            # Wait for all processing threads to complete
            for thread in threads:
                thread.join()
            
            print("Finished speaking text")
    
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
            # Update language based on voice
            if voice.startswith('ru_'):
                self._current_language = "ru"
                self._russian_voice_id = voice
            elif voice.startswith('en_'):
                self._current_language = "en"
                self._english_voice_id = voice
            else:
                for lang in self.model_urls.keys():
                    if voice.startswith(f"{lang}_"):
                        self._current_language = lang
                        break
        
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
            'audio_device': self.audio_device,
            'language': self._current_language
        }
    
    def set_russian_voice(self):
        """
        Set the text-to-speech engine to use a Russian voice if available
        
        Returns:
            True if successful, False if no Russian voice found
        """
        return self.set_language("ru")
        
    def set_english_voice(self):
        """
        Set the text-to-speech engine to use an English voice if available
        
        Returns:
            True if successful, False if no English voice found
        """
        return self.set_language("en")
    
    def set_language(self, language_code):
        """
        Set the text-to-speech engine to use a specific language
        
        Args:
            language_code: Language code (e.g., 'ru', 'en')
            
        Returns:
            True if successful, False otherwise
        """
        if language_code not in self.model_urls:
            print(f"Language {language_code} is not supported")
            return False
            
        # Don't reload if we're already using this language
        if language_code == self._current_language:
            return True
            
        # Load appropriate model
        loading_thread = threading.Thread(target=self.load_model, args=(language_code,), daemon=True)
        loading_thread.start()
        loading_thread.join()  # Wait for model loading to complete
        
        # Verify that it worked
        return self._current_language == language_code
    
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
    
    def get_supported_languages(self):
        """
        Get a list of all supported languages
        
        Returns:
            Dictionary of language codes and names
        """
        return {
            "ru": "Russian",
            "en": "English",
            "de": "German",
            "es": "Spanish",
            "fr": "French"
        }
        
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