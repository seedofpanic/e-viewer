import unittest
import time
import asyncio
import concurrent.futures
from unittest.mock import MagicMock, patch
from PyQt5.QtCore import QObject

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.threading_utils import (
    ThreadingSignals, 
    run_in_thread,
    run_async_in_thread,
    submit_to_thread_pool,
    setup_qt_connect_async
)


class TestThreadingUtils(unittest.TestCase):
    
    def test_run_in_thread(self):
        """Test the run_in_thread decorator"""
        result = []
        
        @run_in_thread
        def test_function():
            time.sleep(0.1)  # Small delay to ensure it runs in separate thread
            result.append("Done")
            
        thread = test_function()
        thread.join()  # Wait for thread to complete
        
        self.assertEqual(result, ["Done"])
        
    def test_run_async_in_thread(self):
        """Test the run_async_in_thread decorator"""
        result = []
        
        @run_async_in_thread()
        async def test_async_function():
            await asyncio.sleep(0.1)  # Small delay
            result.append("Done")
            return "Completed"
            
        thread = test_async_function()
        thread.join()  # Wait for thread to complete
        
        self.assertEqual(result, ["Done"])
        
    def test_run_async_in_thread_with_pool(self):
        """Test the run_async_in_thread decorator with thread pool"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            result = []
            
            @run_async_in_thread(thread_pool=pool)
            async def test_async_function():
                await asyncio.sleep(0.1)  # Small delay
                result.append("Done")
                return "Completed"
                
            future = test_async_function()
            concurrent.futures.wait([future])  # Wait for future to complete
            
            self.assertEqual(result, ["Done"])
            
    def test_submit_to_thread_pool(self):
        """Test submit_to_thread_pool function"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            def test_function(arg1, arg2):
                return arg1 + arg2
                
            future = submit_to_thread_pool(pool, test_function, 5, 7)
            result = future.result()
            
            self.assertEqual(result, 12)
            
    @patch('src.threading_utils.ThreadingSignals')
    def test_setup_qt_connect_async(self, mock_signals_class):
        """Test setup_qt_connect_async function with mocks"""
        # Mock the signals
        mock_signals = MagicMock()
        mock_signals_class.return_value = mock_signals
        
        # Create test callbacks
        callback = MagicMock()
        error_handler = MagicMock()
        finished_callback = MagicMock()
        
        # Create test async function that returns a value
        async def test_async_func():
            return "test_result"
        
        # Set up connections
        signals = setup_qt_connect_async(
            test_async_func,
            callback=callback,
            error_handler=error_handler,
            finished_callback=finished_callback
        )
        
        # Check that signals were connected correctly
        mock_signals.result.connect.assert_called_once_with(callback)
        mock_signals.error.connect.assert_called_once_with(error_handler)
        mock_signals.finished.connect.assert_called_once_with(finished_callback)
        

if __name__ == '__main__':
    unittest.main() 