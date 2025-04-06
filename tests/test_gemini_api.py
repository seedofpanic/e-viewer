import unittest
from unittest.mock import patch, MagicMock, mock_open, ANY
import sys
import os
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gemini_api import GeminiAPI, GoogleAIFileManager


class TestGeminiAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.api_key = "test_api_key"
        self.api = GeminiAPI(self.api_key)
        
    def test_init_with_api_key(self):
        """Test initialization with API key"""
        self.assertEqual(self.api.api_key, self.api_key)
        
    @patch.dict(os.environ, {"GEMINI_API_KEY": "env_api_key"})
    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment"""
        api = GeminiAPI()
        self.assertEqual(api.api_key, "env_api_key")
        
    def test_init_without_api_key(self):
        """Test initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            api = GeminiAPI()
            self.assertEqual(api.api_key, "")
            
    @patch('src.gemini_api.genai')
    def test_configure_with_api_key(self, mock_genai):
        """Test configure method with valid API key"""
        result = self.api.configure()
        mock_genai.configure.assert_called_once_with(api_key=self.api_key)
        self.assertTrue(result)
        
    @patch('src.gemini_api.genai')
    def test_configure_without_api_key(self, mock_genai):
        """Test configure method without API key"""
        self.api.api_key = ""
        result = self.api.configure()
        mock_genai.configure.assert_not_called()
        self.assertFalse(result)
        
    def test_default_generation_config(self):
        """Test default generation config"""
        config = self.api.get_default_generation_config()
        self.assertIsInstance(config, dict)
        self.assertIn("temperature", config)
        self.assertIn("top_p", config)
        self.assertIn("top_k", config)
        
    def test_default_safety_settings(self):
        """Test default safety settings"""
        settings = self.api.get_default_safety_settings()
        self.assertIsInstance(settings, dict)
        self.assertEqual(len(settings), 4)  # Should have 4 harm categories
        
    @patch('src.gemini_api.genai')
    def test_create_model(self, mock_genai):
        """Test create_model method"""
        self.api.create_model("test-model")
        mock_genai.GenerativeModel.assert_called_once_with(
            model_name="test-model",
            generation_config=self.api.get_default_generation_config(),
            safety_settings=self.api.get_default_safety_settings(),
        )
        
    @patch('src.gemini_api.genai')
    def test_create_model_without_api_key(self, mock_genai):
        """Test create_model method without API key"""
        self.api.api_key = ""
        with self.assertRaises(ValueError):
            self.api.create_model()
            
    @patch.object(GeminiAPI, 'create_model')
    async def test_analyze_video(self, mock_create_model):
        """Test analyze_video method"""
        # Mock the model and its response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Video analysis result"
        mock_model.generate_content.return_value = mock_response
        mock_create_model.return_value = mock_model
        
        # Test with minimal parameters
        video_bytes = b"fake video data"
        result = await self.api.analyze_video(video_bytes)
        
        # Verify model was created and generate_content was called with expected args
        mock_create_model.assert_called_once()
        mock_model.generate_content.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result, "Video analysis result")
        
    @patch.object(GeminiAPI, 'create_model')
    async def test_analyze_video_with_custom_prompt(self, mock_create_model):
        """Test analyze_video method with custom prompt"""
        # Mock the model and its response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Custom prompt result"
        mock_model.generate_content.return_value = mock_response
        mock_create_model.return_value = mock_model
        
        # Test with custom prompt
        video_bytes = b"fake video data"
        custom_prompt = "Analyze this video in a funny way"
        result = await self.api.analyze_video(video_bytes, custom_prompt)
        
        # Verify model was created and generate_content was called with expected args
        mock_create_model.assert_called_once()
        mock_model.generate_content.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result, "Custom prompt result")


class TestGoogleAIFileManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.api_key = "test_api_key"
        self.file_manager = GoogleAIFileManager(self.api_key)
        
    @patch('src.gemini_api.genai')
    def test_init(self, mock_genai):
        """Test initialization"""
        self.assertEqual(self.file_manager.api_key, self.api_key)
        mock_genai.configure.assert_called_once_with(api_key=self.api_key)
        
    @patch('builtins.open', new_callable=mock_open, read_data=b"test file content")
    async def test_upload_file(self, mock_file):
        """Test uploadFile method"""
        # Create a temp file path
        temp_file = os.path.join(tempfile.gettempdir(), "test_video.mp4")
        
        # Call the upload method
        result = await self.file_manager.uploadFile(temp_file)
        
        # Check that file was opened
        mock_file.assert_called_once_with(temp_file, 'rb')
        
        # Check results
        self.assertIn("file_id", result)
        self.assertEqual(result["display_name"], "test_video.mp4")
        self.assertEqual(result["mime_type"], "video/mp4")
        
    @patch('builtins.open', new_callable=mock_open, read_data=b"test file content")
    async def test_upload_file_with_options(self, mock_file):
        """Test uploadFile method with custom options"""
        # Create a temp file path
        temp_file = os.path.join(tempfile.gettempdir(), "test_video.mp4")
        
        # Custom options
        options = {
            "displayName": "Custom Name",
            "mimeType": "video/avi"
        }
        
        # Call the upload method
        result = await self.file_manager.uploadFile(temp_file, options)
        
        # Check results
        self.assertEqual(result["display_name"], "Custom Name")
        self.assertEqual(result["mime_type"], "video/avi")
        
    @patch('builtins.open')
    async def test_upload_file_error(self, mock_file):
        """Test uploadFile method with error"""
        # Mock an error when opening the file
        mock_file.side_effect = FileNotFoundError("File not found")
        
        # Create a temp file path
        temp_file = os.path.join(tempfile.gettempdir(), "nonexistent.mp4")
        
        # Call the upload method and expect an exception
        with self.assertRaises(Exception) as context:
            await self.file_manager.uploadFile(temp_file)
            
        # Check error message
        self.assertIn("Error uploading file", str(context.exception))


if __name__ == '__main__':
    unittest.main() 