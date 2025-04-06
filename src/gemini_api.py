import os
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class GoogleAIFileManager:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)

    async def uploadFile(self, file_path, options=None):
        """Upload a file to Google Gemini API"""
        try:
            # Read the file as binary
            with open(file_path, 'rb') as file:
                file_content = file.read()

            # Extract file name for display name if not provided
            if options and 'displayName' not in options:
                display_name = os.path.basename(file_path)
            else:
                display_name = options.get(
                    'displayName', os.path.basename(file_path))

            mime_type = options.get('mimeType', 'video/mp4')

            # Create the file part for Gemini
            file_part = {
                "mime_type": mime_type,
                "data": file_content
            }

            # Return response object with file details
            return {
                "file_id": f"temp_id_{int(time.time())}",
                "display_name": display_name,
                "mime_type": mime_type
            }
        except Exception as e:
            raise Exception(f"Error uploading file: {str(e)}")


class GeminiAPI:
    def __init__(self, api_key=None):
        """Initialize Gemini API with optional API key"""
        # Save provided API key if any
        self.api_key = api_key

        # If not provided, check environment variables, but don't fail if not found
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")

    def configure(self):
        """Configure the Gemini API with the current API key"""
        if not self.api_key:
            # Don't raise exception at configure time - just log a warning
            print(
                "Warning: No Gemini API key configured. API calls will require an API key from UI.")
            return False

        genai.configure(api_key=self.api_key)
        return True

    def get_default_generation_config(self):
        """Get default generation configuration"""
        return {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 0,
        }

    def get_default_safety_settings(self):
        """Get default safety settings (permissive)"""
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def create_model(self, model_name="gemini-2.0-flash"):
        """Create a Gemini model with default settings"""
        # Check if we have an API key - if not, raise an informative error
        if not self.api_key:
            raise ValueError(
                "No API key available. Please enter a valid Gemini API key in the application UI.")

        self.configure()
        return genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.get_default_generation_config(),
            safety_settings=self.get_default_safety_settings(),
        )

    async def analyze_video(self, video_bytes, custom_prompt=None, context=None):
        """Analyze a video with Gemini

        Args:
            video_bytes: Raw bytes of the video file
            custom_prompt: Optional custom prompt to override default, can be a string or a dictionary
            context: Optional context from previous conversations

        Returns:
            Text response from Gemini
        """
        try:
            if not video_bytes:
                return "Error: No video data provided"

            if not self.api_key:
                return "No API key available. Please enter a valid Gemini API key in the application UI."

            model = self.create_model()

            # Correct structure: a dictionary with mime_type and data
            video_part = {
                "mime_type": "video/mp4",
                "data": video_bytes
            }

            if not custom_prompt:
                prompt = "Please analyze this video and provide a detailed description and any observations about what you see. Analyze any activities, applications, text content visible in the recording."
            else:
                # Handle dictionary format prompt
                if isinstance(custom_prompt, dict):
                    # Combine character, task, and style guidelines into a structured prompt
                    prompt = f"{custom_prompt.get('character', '')}\n\n"
                    prompt += f"{custom_prompt.get('task', '')}\n\n"
                    prompt += f"Style guidelines: {custom_prompt.get('style_guidelines', '')}\n\n"

                    if 'additional_instructions' in custom_prompt:
                        prompt += f"Additional instructions: {custom_prompt.get('additional_instructions', '')}\n\n"

                    if 'output_format' in custom_prompt:
                        prompt += f"Output format: {custom_prompt.get('output_format', '')}"
                else:
                    # Use the prompt as is if it's a string
                    prompt = custom_prompt

            # Start a chat if we have previous context
            if context:
                chat = model.start_chat(history=[])

                # If we have a previous conversation, provide it as context
                if "last_response" in context:
                    # Add previous context to the chat history
                    chat.history.append({
                        "role": "user",
                        "parts": ["Here's a video I showed you previously"]
                    })
                    chat.history.append({
                        "role": "model",
                        "parts": [context["last_response"]]
                    })

                # Add new query with the video
                response = chat.send_message([prompt, video_part])
                return response.text
            else:
                # If no context, use standard generate_content
                response = model.generate_content([prompt, video_part])
                return response.text
        except Exception as e:
            error_message = f"Error analyzing video: {str(e)}"
            print(f"GEMINI API ERROR: {error_message}")
            return error_message
