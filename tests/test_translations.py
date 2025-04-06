import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.translations import TRANSLATIONS


class TestTranslations(unittest.TestCase):
    
    def test_translations_structure(self):
        """Test the structure of the translations dict"""
        # Check that the translations dictionary has entries for English and Russian
        self.assertIn("en", TRANSLATIONS)
        self.assertIn("ru", TRANSLATIONS)
        
    def test_english_translations(self):
        """Test that English translations contain expected keys"""
        en_translations = TRANSLATIONS["en"]
        
        # Check essential keys
        essential_keys = [
            "window_title", 
            "status_ready", 
            "recording_controls",
            "buffer_size",
            "start_recording",
            "stop_recording",
            "analysis_controls",
            "prompt",
            "run_analysis",
            "hotkey",
            "tts_volume",
            "tts_speed",
            "tts_voice"
        ]
        
        for key in essential_keys:
            self.assertIn(key, en_translations)
            self.assertIsInstance(en_translations[key], str)
            self.assertTrue(len(en_translations[key]) > 0)
            
    def test_russian_translations(self):
        """Test that Russian translations contain expected keys"""
        ru_translations = TRANSLATIONS["ru"]
        
        # Check essential keys
        essential_keys = [
            "window_title", 
            "status_ready", 
            "recording_controls",
            "buffer_size",
            "start_recording",
            "stop_recording",
            "analysis_controls",
            "prompt",
            "run_analysis",
            "hotkey",
            "tts_volume",
            "tts_speed",
            "tts_voice"
        ]
        
        for key in essential_keys:
            self.assertIn(key, ru_translations)
            self.assertIsInstance(ru_translations[key], str)
            self.assertTrue(len(ru_translations[key]) > 0)
            
    def test_translation_key_parity(self):
        """Test that English and Russian translations have the same keys"""
        en_keys = set(TRANSLATIONS["en"].keys())
        ru_keys = set(TRANSLATIONS["ru"].keys())
        
        self.assertEqual(en_keys, ru_keys, 
                        f"English and Russian translations should have the same keys. "
                        f"Missing in English: {ru_keys - en_keys}. "
                        f"Missing in Russian: {en_keys - ru_keys}.")
    
    def test_format_strings(self):
        """Test that format strings in translations work properly"""
        # Test error message format string
        error_msg = "Test Error"
        en_formatted = TRANSLATIONS["en"]["status_error"].format(error_msg)
        ru_formatted = TRANSLATIONS["ru"]["status_error"].format(error_msg)
        
        self.assertIn(error_msg, en_formatted)
        self.assertIn(error_msg, ru_formatted)


if __name__ == '__main__':
    unittest.main() 