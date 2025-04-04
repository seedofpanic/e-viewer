import sys
import concurrent.futures
from PyQt5.QtWidgets import QApplication
from screen_recorder_app import ScreenRecorderApp

def main():
    print("Starting Screen Recorder Application")
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create thread pool for background operations
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    # Create and initialize the main app
    main_window = ScreenRecorderApp()
    main_window.thread_pool = thread_pool
    
    # Show the application window
    print("Showing main window")
    main_window.show()
    
    # Run application
    print("Starting event loop")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 