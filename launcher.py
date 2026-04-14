"""
СІМІ Dashboard — launcher
Запускає Streamlit і автоматично відкриває браузер.
"""
import subprocess
import sys
import os
import time
import webbrowser
import threading

def open_browser():
    time.sleep(2.5)
    webbrowser.open("http://localhost:8501")

def main():
    # Шлях до app.py поруч з лончером
    base = os.path.dirname(os.path.abspath(__file__))
    app  = os.path.join(base, "app.py")

    threading.Thread(target=open_browser, daemon=True).start()

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app,
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ])

if __name__ == "__main__":
    main()
