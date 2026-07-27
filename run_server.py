"""
EmotionAI Application Runner
High-performance launcher script for local development and production.
"""
import sys
import os

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add working directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == "__main__":
    print("=" * 60)
    print(" [EmotionAI] Real-Time Facial Emotion Recognition")
    print(" Access application at: http://localhost:5000")
    print(" Engine: TensorFlow Mini-Xception + OpenCV + Client REST API")
    print("=" * 60)

    try:
        from waitress import serve
        print(" [INFO] Serving with Waitress Multi-Threaded WSGI Server...")
        serve(app, host='0.0.0.0', port=5000, threads=6)
    except ImportError:
        print(" [INFO] Waitress not installed, running with standard Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
