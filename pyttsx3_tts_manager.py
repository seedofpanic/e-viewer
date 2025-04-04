"""
Text-to-Speech Manager using pyttsx3 library.
This is a simpler alternative to the Silero TTS implementation.
"""

import pyttsx3
import threading
import queue
import time
import os
from pathlib import Path

class PyTTSX3Manager:
    def __init__(self):
        """Initialize the pyttsx3 text-to-speech manager"""
        # Engine properties
        self.rate = 150  # Default speech rate (words per minute)
        self.volume = 1.0  # Default volume level (0.0 to 1.0)
        self.current_voice = None
        
        # Create a queue for speech tasks
        self._speech_queue = queue.Queue()
        self._is_speaking = False
        self._stop_requested = False
        self._engine = None
        self._voices = []
        self._russian_voice_id = None
        
        # Start the speech processing thread
        self._speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self._speech_thread.start()
        
        # Initialize the engine (non-blocking)
        self._init_engine()
    
    def _init_engine(self):
        """Initialize the pyttsx3 engine"""
        try:
            self._engine = pyttsx3.init()
            
            # Get available voices
            self._voices = self._engine.getProperty('voices')
            
            # Set default voice (first voice)
            if self._voices:
                self.current_voice = self._voices[0].id
                self._engine.setProperty('voice', self.current_voice)
            
            # Set initial rate and volume
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            
            # Find a Russian voice if available
            for voice in self._voices:
                if 'ru' in voice.id.lower() or 'russian' in voice.name.lower():
                    self._russian_voice_id = voice.id
                    self.current_voice = self._russian_voice_id  # Set Russian as default
                    break
            
            print(f"pyttsx3 TTS engine initialized with {len(self._voices)} voices")
            if self._russian_voice_id:
                print(f"Russian voice found: {self._russian_voice_id}")
                
        except Exception as e:
            print(f"Error initializing pyttsx3 engine: {str(e)}")
    
    def _process_speech_queue(self):
        """Process speech tasks from the queue"""
        while True:
            try:
                # Get the next speech task
                task = self._speech_queue.get()
                
                # Skip if engine not ready
                if not self._engine:
                    print("Engine not ready, skipping speech task")
                    self._speech_queue.task_done()
                    continue
                
                # Extract task parameters
                text, rate, volume, voice, is_save, filename = task
                
                # Update properties
                if rate is not None:
                    self._engine.setProperty('rate', rate)
                else:
                    self._engine.setProperty('rate', self.rate)
                    
                if volume is not None:
                    self._engine.setProperty('volume', volume)
                else:
                    self._engine.setProperty('volume', self.volume)
                    
                selected_voice = voice if voice is not None else self.current_voice
                
                # Check for Russian text
                has_cyrillic = any(ord('а') <= ord(c) <= ord('я') or ord('А') <= ord(c) <= ord('Я') for c in text)
                if has_cyrillic and self._russian_voice_id and selected_voice != self._russian_voice_id:
                    selected_voice = self._russian_voice_id
                
                if selected_voice:
                    self._engine.setProperty('voice', selected_voice)
                
                # Set speaking flag
                self._is_speaking = True
                
                # Process task
                if is_save and filename:
                    # Save to file
                    try:
                        self._engine.save_to_file(text, filename)
                        self._engine.runAndWait()
                        print(f"Speech saved to file: {filename}")
                    except Exception as e:
                        print(f"Error saving speech to file: {str(e)}")
                else:
                    # Speak
                    try:
                        self._engine.say(text)
                        self._engine.runAndWait()
                        print("Finished speaking text")
                    except Exception as e:
                        print(f"Error speaking text: {str(e)}")
                
                # Reset flag and mark task as done
                self._is_speaking = False
                self._speech_queue.task_done()
                
                # Add a small delay to ensure the engine has time to reset
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in speech thread: {str(e)}")
                time.sleep(0.5)  # Delay to prevent tight loop if there's an error
    
    def speak_text(self, text, rate=None, volume=None, voice=None):
        """Convert text to speech using pyttsx3
        
        Args:
            text: Text to be spoken
            rate: Speed of speech (words per minute)
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
            rate: Speed of speech (words per minute)
            volume: Volume level (0.0 to 1.0)
            voice: Voice ID to use for speech
            
        Returns:
            Dictionary of current TTS properties
        """
        if rate is not None:
            self.rate = rate
        if volume is not None:
            self.volume = volume
        if voice is not None and self._voices and voice in [v.id for v in self._voices]:
            self.current_voice = voice
        
        # Return current properties
        return self.get_tts_properties()
    
    def get_tts_properties(self):
        """
        Get current text-to-speech properties
        
        Returns:
            Dictionary of current TTS properties
        """
        return {
            'rate': self.rate,
            'volume': self.volume,
            'voice': self.current_voice
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
        else:
            print("No Russian voice found. Using default voice.")
            return False
    
    def get_available_voices(self):
        """
        Get a list of all available TTS voices
        
        Returns:
            List of dictionaries with voice information
        """
        if not self._voices:
            return []
            
        voice_list = []
        for voice in self._voices:
            # Try to determine language from the voice ID or name
            language = "unknown"
            if "ru" in voice.id.lower() or "russian" in voice.name.lower():
                language = "ru"
            elif "en" in voice.id.lower() or "english" in voice.name.lower():
                language = "en"
            
            voice_info = {
                'id': voice.id,
                'name': voice.name,
                'languages': [language],
                'gender': voice.gender if hasattr(voice, 'gender') else "unknown",
                'age': voice.age if hasattr(voice, 'age') else "unknown"
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