import unittest
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from src.hotkey_input import HotkeyInput


# Create a QApplication instance for the tests
app = QApplication.instance()
if app is None:
    # Use minimal platform plugin or respect the environment variable
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = QApplication([])


class TestHotkeyInput(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.hotkey_input = HotkeyInput()
        
    def test_initialization(self):
        """Test that HotkeyInput initializes with correct properties"""
        self.assertTrue(self.hotkey_input.isReadOnly())
        self.assertEqual(self.hotkey_input.maximumHeight(), 30)
        self.assertEqual(self.hotkey_input.placeholderText(), "Click here to set hotkey...")
        self.assertFalse(self.hotkey_input.is_listening)
        
    def test_toggle_listening_mode(self):
        """Test toggling between listening and non-listening modes"""
        # Initial state
        self.assertFalse(self.hotkey_input.is_listening)
        
        # Toggle to listening mode
        self.hotkey_input.toggle_listening_mode()
        self.assertTrue(self.hotkey_input.is_listening)
        self.assertEqual(self.hotkey_input.placeholderText(), "Press key combination...")
        
        # Toggle back to non-listening mode
        self.hotkey_input.toggle_listening_mode()
        self.assertFalse(self.hotkey_input.is_listening)
        self.assertEqual(self.hotkey_input.placeholderText(), "Click here to set hotkey...")
        
    def test_mouse_press_toggles_listening(self):
        """Test that mouse press toggles listening mode"""
        # Mock the toggle_listening_mode method to check if it's called
        self.hotkey_input.toggle_listening_mode = MagicMock()
        
        # Simulate left mouse button click
        QTest.mouseClick(self.hotkey_input.viewport(), Qt.LeftButton)
        
        # Check if toggle_listening_mode was called
        self.hotkey_input.toggle_listening_mode.assert_called_once()
        
    def test_key_press_in_listening_mode(self):
        """Test handling key press events in listening mode"""
        # Set up a signal spy to catch the hotkeyChanged signal
        signal_spy = MagicMock()
        self.hotkey_input.hotkeyChanged.connect(signal_spy)
        
        # Enable listening mode
        self.hotkey_input.is_listening = True
        
        # Create a key event with Ctrl+A
        ctrl_a_event = MagicMock()
        ctrl_a_event.key.return_value = Qt.Key_A
        ctrl_a_event.modifiers.return_value = Qt.ControlModifier
        
        # QKeySequence mocking is a bit tricky, so we'll patch it
        with patch('src.hotkey_input.QKeySequence') as mock_key_sequence:
            mock_key_sequence.return_value.toString.return_value = "A"
            
            # Simulate key press event
            self.hotkey_input.keyPressEvent(ctrl_a_event)
            
            # Check if the text was updated correctly
            self.assertEqual(self.hotkey_input.toPlainText(), "Ctrl+A")
            
            # Check if signal was emitted with correct value
            signal_spy.assert_called_once_with("Ctrl+A")
            
            # Check if listening mode was turned off
            self.assertFalse(self.hotkey_input.is_listening)
            
    def test_key_press_ignored_when_not_listening(self):
        """Test that key presses are ignored when not in listening mode"""
        # Ensure not in listening mode
        self.hotkey_input.is_listening = False
        self.hotkey_input.setText("Existing Hotkey")
        
        # Create a key event
        key_event = MagicMock()
        
        # Process the event
        self.hotkey_input.keyPressEvent(key_event)
        
        # Text should remain unchanged
        self.assertEqual(self.hotkey_input.toPlainText(), "Existing Hotkey")
        
    def test_init_emits_current_hotkey(self):
        """Test that init method emits the current hotkey"""
        # Set up a signal spy
        signal_spy = MagicMock()
        self.hotkey_input.hotkeyChanged.connect(signal_spy)
        
        # Set some text and call init
        self.hotkey_input.setText("Ctrl+B")
        self.hotkey_input.init()
        
        # Check if signal was emitted with correct value
        signal_spy.assert_called_once_with("Ctrl+B")


if __name__ == '__main__':
    unittest.main() 