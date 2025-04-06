import os
import time
import cv2
import numpy as np
import tempfile
from collections import deque
import threading
import concurrent.futures
import pynvml
import asyncio
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal
from gemini_api import GeminiAPI, GoogleAIFileManager
from prompts import get_prompt, get_random_prompt
from threading_utils import ThreadingSignals, run_async_in_thread

# Add a Qt-friendly signal class


class ScreenRecorderSignals(QObject):
    frame_captured = pyqtSignal(object)
    processing_complete = pyqtSignal(object)
    error_occurred = pyqtSignal(str)


class ScreenRecorder:
    def __init__(self):
        # Initialize NVML (NVIDIA Management Library)
        self.nvml_initialized = False
        self.has_nvenc = False

        # Add signals for cross-thread communication
        self.signals = ScreenRecorderSignals()

        # Get screen dimensions in the main thread
        self.screen = QApplication.primaryScreen()
        self.screen_size = self.screen.size()
        self.orig_width, self.orig_height = self.screen_size.width(), self.screen_size.height()

        # Set target resolution to 720p while maintaining aspect ratio
        self.target_height = 720
        if self.orig_width > self.orig_height:
            # Landscape orientation
            self.target_width = int(
                self.orig_width * (self.target_height / self.orig_height))
            # Ensure width is even for video encoding compatibility
            self.target_width = self.target_width - (self.target_width % 2)
        else:
            # Portrait orientation
            self.target_width = 720
            self.target_height = int(
                self.orig_height * (self.target_width / self.orig_width))
            # Ensure height is even for video encoding compatibility
            self.target_height = self.target_height - (self.target_height % 2)

        self.frame_size = (self.target_width, self.target_height)
        print(f"Original resolution: {self.orig_width}x{self.orig_height}")
        print(f"Recording at 720p: {self.target_width}x{self.target_height}")

        # Create internal thread pool
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
            print("NVIDIA Management Library initialized successfully")

            # Check for NVENC support
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                # Get the first GPU
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                device_name = pynvml.nvmlDeviceGetName(handle)
                print(f"GPU detected: {device_name}")
                self.has_nvenc = True

                # On Windows, XVID is more reliable than h264/avc1
                self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
                print("Using NVIDIA hardware acceleration with XVID codec")
            else:
                print("No NVIDIA GPUs detected")
                self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        except Exception as e:
            print(f"Failed to initialize NVIDIA Management Library: {e}")
            print("Falling back to software encoding")
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Initialize screen capture
        self.recording = False
        self.buffer_size = 10  # 10 seconds of footage
        # Target frames per second (reduced from 30 to 15 for lower resource usage)
        self.fps = 10
        self.frame_buffer = deque(maxlen=self.buffer_size * self.fps)
        self.lock = threading.Lock()
        self.capture_thread = None
        self.processing_futures = []

        # Initialize video writer properties
        self.output_file = None
        self.video_writer = None

        # Get API key from environment variables
        self.gemini_api = GeminiAPI()

        # Context storage for persistent conversations
        self.context_storage = {}

    def start_capture(self, api_key=None):
        """Start capturing screen frames"""
        if self.recording:
            return

        # Update the Gemini API instance with the provided API key
        if api_key:
            self.gemini_api.api_key = api_key
            print(f"Updated Gemini API with key from UI")

        self.recording = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print("Screen capture started")

    def stop_capture(self):
        """Stop capturing screen frames"""
        self.recording = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
        print("Screen capture stopped")

        # Cancel any pending futures
        for future in self.processing_futures:
            if not future.done():
                future.cancel()
        self.processing_futures = []

    def _capture_loop(self):
        """Main capture loop that runs in a separate thread"""
        capture_start_time = time.time()
        frames_captured = 0

        try:
            while self.recording:
                loop_start = time.time()

                # Capture screen using Qt in the main thread
                pixmap = self.screen.grabWindow(0)
                image = pixmap.toImage()

                # Convert QImage to numpy array for OpenCV
                width = image.width()
                height = image.height()
                ptr = image.constBits()
                ptr.setsize(image.byteCount())
                arr = np.array(ptr).reshape(height, width, 4)  # Format is RGBA

                # Convert RGBA to BGR (OpenCV format) and resize in a thread pool
                future = self.thread_pool.submit(self._process_frame, arr)
                self.processing_futures.append(future)
                self.processing_futures = [
                    f for f in self.processing_futures if not f.done()]

                frames_captured += 1

                # Calculate sleep time to maintain target FPS
                process_time = time.time() - loop_start
                sleep_time = max(0, 1.0/self.fps - process_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"Error in capture loop: {e}")
            self.signals.error_occurred.emit(f"Capture error: {str(e)}")

    def _process_frame(self, frame_arr):
        """Process captured frame in thread pool"""
        try:
            # Convert RGBA to BGR (OpenCV format)
            # The input might actually be BGRA instead of RGBA, causing the blue tint
            # Try BGR conversion instead
            frame = cv2.cvtColor(frame_arr, cv2.COLOR_BGRA2BGR)

            # Resize to 720p resolution
            frame = cv2.resize(frame, (self.target_width, self.target_height))

            # Store frame in buffer
            with self.lock:
                self.frame_buffer.append((time.time(), frame))

            return frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def save_last_seconds(self, output_path=None):
        """Save the last N seconds from the buffer to a video file"""
        if not self.frame_buffer:
            print("No frames in buffer to save")
            return False

        if output_path is None:
            # Generate a default filename if none provided
            # .avi for XVID codec
            output_path = f"screen_capture_{int(time.time())}.avi"
        elif not output_path.lower().endswith('.avi'):
            # Ensure correct extension for XVID codec
            output_path = os.path.splitext(output_path)[0] + '.avi'

        # Create a future for the save operation
        save_future = self.thread_pool.submit(self._save_video, output_path)
        self.processing_futures.append(save_future)

        return save_future

    def _save_video(self, output_path):
        """Save video in a worker thread"""
        with self.lock:
            if not self.frame_buffer:
                return False

            # Copy frames to avoid holding the lock too long
            frames_to_save = list(self.frame_buffer)

        # Get frame dimensions from the first frame
        _, first_frame = frames_to_save[0]
        h, w = first_frame.shape[:2]

        try:
            # Create video writer with MJPG codec which handles pts better
            video_writer = cv2.VideoWriter(output_path,
                                           cv2.VideoWriter_fourcc(*'MJPG'),
                                           self.fps, (w, h))

            if not video_writer.isOpened():
                print("Failed to open video writer.")
                return False

            # Write frames sequentially to avoid pts issues
            for _, frame in frames_to_save:
                video_writer.write(frame)

            video_writer.release()
            print(f"Last {self.buffer_size} seconds saved to {output_path}")

            # Signal completion
            self.signals.processing_complete.emit(
                {"type": "save", "path": output_path})
            return True
        except Exception as e:
            print(f"Error saving video: {e}")
            self.signals.error_occurred.emit(f"Save error: {str(e)}")
            return False

    def save_buffer_to_temp_file(self):
        """Save the current buffer to a temporary file and return the path"""
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
        temp_file_path = temp_file.name
        temp_file.close()

        # Get all frames from buffer
        with self.lock:
            if not self.frame_buffer:
                return None

            # Copy frames to avoid holding the lock too long
            frames_to_save = list(self.frame_buffer)

        # Get frame dimensions from the first frame
        _, first_frame = frames_to_save[0]
        h, w = first_frame.shape[:2]

        try:
            # Use optimized writer with hardware acceleration if available
            if self.has_nvenc:
                # NVIDIA hardware acceleration
                print("Using NVIDIA hardware acceleration for saving")

            # Create video writer with XVID codec - ensure unique timestamp for each temp file
            timestamp = int(time.time() * 1000)
            unique_temp_path = f"{os.path.splitext(temp_file_path)[0]}_{timestamp}.avi"

            # Use MJPG codec which is more reliable than XVID for presentation timestamps
            video_writer = cv2.VideoWriter(unique_temp_path,
                                           cv2.VideoWriter_fourcc(*'MJPG'),
                                           self.fps, (w, h))

            if not video_writer.isOpened():
                print("Failed to open video writer.")
                if os.path.exists(unique_temp_path):
                    os.unlink(unique_temp_path)
                return None

            # Process frames sequentially to avoid pts conflicts
            # This is slower but more reliable than parallel processing
            for _, frame in frames_to_save:
                video_writer.write(frame)

            # Properly release writer and ensure all frames are written
            video_writer.release()

            # Wait a moment to ensure file is properly closed
            time.sleep(0.1)

            print(f"Buffer saved to temporary file: {unique_temp_path}")
            return unique_temp_path

        except Exception as e:
            print(f"Error saving buffer to file: {e}")
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as delete_error:
                    print(f"Could not delete temp file: {delete_error}")
            return None

    def set_buffer_size(self, seconds):
        """
        Set the buffer size in seconds

        Args:
            seconds: Number of seconds to keep in the buffer

        Returns:
            None
        """
        if seconds < 1:
            seconds = 1

        # Update buffer size
        self.buffer_size = seconds

        # Update the frame buffer deque with new maxlen
        with self.lock:
            # Create a new deque with the updated maxlen
            new_buffer = deque(maxlen=self.buffer_size * self.fps)

            # Copy over existing frames if any
            if self.frame_buffer:
                # Only copy as many frames as will fit in the new buffer
                frames_to_copy = list(
                    self.frame_buffer)[-min(len(self.frame_buffer), new_buffer.maxlen):]
                for frame in frames_to_copy:
                    new_buffer.append(frame)

            # Replace the old buffer with the new one
            self.frame_buffer = new_buffer

        print(
            f"Buffer size changed to {seconds} seconds ({self.frame_buffer.maxlen} frames)")

    def minimize_video(self, input_path, output_path=None, target_size_mb=10, maintain_aspect_ratio=True):
        """
        Minimize the size of a video file by reducing resolution and bitrate

        Args:
            input_path: Path to the input video
            output_path: Path for the output video (if None, creates a temp file)
            target_size_mb: Target size in megabytes
            maintain_aspect_ratio: Whether to maintain the aspect ratio when resizing

        Returns:
            Path to the minimized video file
        """
        if output_path is None:
            # Create a temporary file with .avi extension with unique timestamp
            timestamp = int(time.time() * 1000)
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=f'_{timestamp}.avi')
            output_path = temp_file.name
            temp_file.close()
        elif not output_path.lower().endswith('.avi'):
            # Ensure correct extension for codec
            output_path = os.path.splitext(output_path)[0] + '.avi'

        try:
            # Open the input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error opening video file: {input_path}")
                return None

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate target bitrate to achieve desired size
            target_size_bytes = target_size_mb * 1024 * 1024
            duration = frame_count / fps
            target_bitrate = int((target_size_bytes * 8) / duration)

            # Determine optimal resolution
            # We'll scale down the resolution if it's too high
            max_dimension = 640  # Reduced from 720p to 640p for faster processing
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
            else:
                # Keep the resolution if it's already small enough
                new_width, new_height = width, height

            # Ensure dimensions are even (required for some codecs)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)

            # Use MJPG codec which is more reliable
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')

            out = cv2.VideoWriter(output_path, fourcc,
                                  fps, (new_width, new_height))

            if not out.isOpened():
                print(f"Error creating output video file. Trying alternative codec.")
                # Try DIVX codec as a fallback
                divx_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter(
                    output_path, divx_fourcc, fps, (new_width, new_height))

                if not out.isOpened():
                    print(f"Failed to create output video with any codec")
                    return None

            # Process frames sequentially to avoid presentation timestamp issues
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if new_width != width or new_height != height:
                    # Use INTER_LINEAR for faster resizing
                    resized = cv2.resize(
                        frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    out.write(resized)
                else:
                    out.write(frame)

            # Release resources
            cap.release()
            out.release()

            # Ensure the file is fully written and closed
            time.sleep(0.1)

            # Check if output file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Minimized video saved to: {output_path}")
                print(
                    f"Original size: {os.path.getsize(input_path) / (1024*1024):.2f} MB")
                print(
                    f"New size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                return output_path
            else:
                print("Failed to create output video file")
                return None

        except Exception as e:
            print(f"Error minimizing video: {e}")
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except Exception as delete_error:
                    print(f"Could not delete output file: {delete_error}")
            return None

    def save_last_seconds_minimized(self, output_path=None, target_size_mb=10):
        """
        Save the last N seconds from the buffer to a minimized video file

        Args:
            output_path: Path for the output video
            target_size_mb: Target size in megabytes

        Returns:
            Success status (bool)
        """
        # First save at normal quality
        temp_output = tempfile.NamedTemporaryFile(
            delete=False, suffix='.avi').name
        success = self.save_last_seconds(temp_output)
        if not success:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
            return False

        # Then minimize the video
        if output_path is None:
            output_path = f"minimized_capture_{int(time.time())}.avi"

        minimized_path = self.minimize_video(
            temp_output, output_path, target_size_mb)

        # Clean up temporary file
        if os.path.exists(temp_output):
            os.unlink(temp_output)

        return minimized_path is not None

    def _encode_frames_to_video(self, frames):
        """
        Encode a list of frames into an in-memory video

        Args:
            frames: List of (timestamp, frame) tuples

        Returns:
            Video bytes or None if encoding failed
        """
        if not frames:
            return None

        # Get dimensions from first frame
        first_frame = frames[0][1]
        frame_height, frame_width = first_frame.shape[:2]

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as temp_buffer:
            temp_path = temp_buffer.name

        try:
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            in_memory_writer = cv2.VideoWriter(
                temp_path, fourcc, self.fps,
                (frame_width, frame_height)
            )

            # Write frames
            for _, frame in frames:
                in_memory_writer.write(frame)

            # Release the writer to ensure data is flushed
            in_memory_writer.release()

            # Read the file into memory
            with open(temp_path, 'rb') as f:
                video_bytes = f.read()

            return video_bytes

        except Exception as e:
            print(f"Error encoding video: {str(e)}")
            return None
        finally:
            # Delete the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def _prepare_video_frames(self, max_age_seconds=5.0, max_dimension=640):
        """
        Extract recent frames from buffer and resize them if needed

        Args:
            max_age_seconds: Only include frames from the last N seconds
            max_dimension: Maximum dimension for frame resize

        Returns:
            List of (timestamp, frame) tuples or None if no frames available
        """
        with self.lock:
            if not self.frame_buffer:
                return None

            # Get only frames from the specified time window
            current_time = time.time()
            frames_to_process = []
            for timestamp, frame in self.frame_buffer:
                if current_time - timestamp <= max_age_seconds:
                    frames_to_process.append((timestamp, frame))

            if not frames_to_process:
                return None

        # Get frame dimensions from the first frame
        _, first_frame = frames_to_process[0]
        h, w = first_frame.shape[:2]

        # Determine optimal resolution for smaller data size
        if w > max_dimension or h > max_dimension:
            if w > h:
                new_width = max_dimension
                new_height = int(h * (max_dimension / w))
            else:
                new_height = max_dimension
                new_width = int(w * (max_dimension / h))

            # Ensure dimensions are even (required for encoding)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)

            # Resize all frames
            resized_frames = []
            for timestamp, frame in frames_to_process:
                resized = cv2.resize(
                    frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                resized_frames.append((timestamp, resized))
            return resized_frames

        return frames_to_process
