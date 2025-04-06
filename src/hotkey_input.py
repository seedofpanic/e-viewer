from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence, QColor
from PyQt5.QtWidgets import QTextEdit

# Custom hotkey input widget


class HotkeyInput(QTextEdit):
    hotkeyChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(30)
        self.setPlaceholderText("Click here to set hotkey...")
        self.is_listening = False
        self.normal_background = self.palette().color(self.viewport().backgroundRole())
        self.listening_background = QColor(
            240, 240, 255)  # Light blue when listening

    def init(self):
        self.hotkeyChanged.emit(self.toPlainText())

    def mousePressEvent(self, event):
        """Handle mouse clicks to toggle listening mode"""
        if event.button() == Qt.LeftButton:
            self.toggle_listening_mode()
        super().mousePressEvent(event)

    def toggle_listening_mode(self):
        """Toggle between active and inactive state"""
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.setPlaceholderText("Press key combination...")
            self.setText("")
            self.viewport().setStyleSheet(
                f"background-color: {self.listening_background.name()};")
        else:
            self.setPlaceholderText("Click here to set hotkey...")
            self.viewport().setStyleSheet(
                f"background-color: {self.normal_background.name()};")

    def keyPressEvent(self, event):
        # Only process keys when in listening mode
        if not self.is_listening:
            return

        # Skip standalone modifiers
        if event.key() in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
            return

        key_text = ""
        modifiers = event.modifiers()

        # Add modifier keys
        if modifiers & Qt.ControlModifier:
            key_text += "Ctrl+"
        if modifiers & Qt.AltModifier:
            key_text += "Alt+"
        if modifiers & Qt.ShiftModifier:
            key_text += "Shift+"
        if modifiers & Qt.MetaModifier:
            key_text += "Meta+"

        # Add the main key
        key = event.key()
        key_name = QKeySequence(key).toString()

        # Combine modifiers and key
        key_text += key_name

        # Update text field
        self.setText(key_text)

        # Emit signal with new hotkey
        self.hotkeyChanged.emit(key_text)

        # Exit listening mode after a key is set
        self.is_listening = False
        self.viewport().setStyleSheet(
            f"background-color: {self.normal_background.name()};")

        # Prevent default handling
        event.accept()
