from flask import Flask, render_template, request, Response, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import cv2
from mtcnn import MTCNN
import threading
import time
from datetime import datetime

# Flask App Initialize
app = Flask(__name__)

# Model Load
model = load_model("models/emotion_model.keras")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion emoji mapping
emotion_emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

# Initialize MTCNN for accurate face detection
detector = MTCNN()

# Store recent emotion detections for history
emotion_history = []
emotion_history_lock = threading.Lock()

# Live detection stats
live_stats = {
    'current_emotion': None,
    'confidence': 0,
    'faces_count': 0,
    'all_scores': {}
}
live_stats_lock = threading.Lock()


# Camera manager for proper resource handling
class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.camera is None or not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    return False
            self.is_running = True
            return True

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.camera is not None and self.camera.isOpened():
                self.camera.release()
                self.camera = None

    def read_frame(self):
        with self.lock:
            if self.camera is not None and self.camera.isOpened():
                success, frame = self.camera.read()
                if success:
                    return frame
        return None


camera_manager = CameraManager()


# Route for Home Page
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            try:
                file_path = os.path.join("static/uploads", file.filename)
                file.save(file_path)

                frame = cv2.imread(file_path)
                if frame is None:
                    error = "Error: Could not open or find the image."
                    return render_template("index.html", result=None, error=error)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if len(faces) == 0:
                    error = "No face detected in the image. Please upload a clear photo with a visible face."
                    return render_template("index.html", result=None, error=error)

                all_faces_data = []

                for face in faces:
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)

                    sub_face_img = gray[y:y + h, x:x + w]
                    if sub_face_img.size == 0:
                        continue

                    resized = cv2.resize(sub_face_img, (48, 48))
                    normalize = resized / 255.0
                    reshaped = np.reshape(normalize, (1, 48, 48, 1))
                    result = model.predict(reshaped, verbose=0)
                    label = np.argmax(result, axis=1)[0]
                    confidence = float(np.max(result) * 100)

                    # Get all emotion scores for this face
                    scores = {emotion_labels[i]: float(result[0][i] * 100) for i in range(len(emotion_labels))}

                    all_faces_data.append({
                        'emotion': emotion_labels[label],
                        'emoji': emotion_emojis[emotion_labels[label]],
                        'confidence': round(confidence, 1),
                        'scores': {k: round(v, 1) for k, v in scores.items()}
                    })

                    # Draw on frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (138, 43, 226), 2)
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (138, 43, 226), -1)
                    text = f"{emotion_labels[label]} {confidence:.0f}%"
                    cv2.putText(frame, text, (x + 5, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                output_image_path = os.path.join("static/uploads", "output_" + file.filename)
                cv2.imwrite(output_image_path, frame)

                # Add to history
                with emotion_history_lock:
                    emotion_history.append({
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'faces': len(all_faces_data),
                        'primary_emotion': all_faces_data[0]['emotion'] if all_faces_data else 'unknown',
                        'primary_emoji': all_faces_data[0]['emoji'] if all_faces_data else '❓'
                    })
                    if len(emotion_history) > 20:
                        emotion_history.pop(0)

                return render_template("index.html",
                                       result=all_faces_data[0]['emotion'] if all_faces_data else None,
                                       image="output_" + file.filename,
                                       faces_data=all_faces_data,
                                       faces_count=len(all_faces_data))

            except UnidentifiedImageError:
                error = "Invalid image format. Please upload a valid image file."
            except Exception as e:
                error = f"An error occurred: {str(e)}"

    return render_template("index.html", result=None, error=error)


# Route for Live Emotion Detection
@app.route("/live")
def live_emotion_detection():
    return render_template("live.html")


# Route to start the camera
@app.route('/start_feed')
def start_feed():
    success = camera_manager.start()
    return jsonify({"status": "started" if success else "failed"})


# Route to stop the camera
@app.route('/stop_feed')
def stop_feed():
    camera_manager.stop()
    return jsonify({"status": "stopped"})


# Route to get live stats
@app.route('/live_stats')
def get_live_stats():
    with live_stats_lock:
        return jsonify(live_stats)


# Route to get emotion history
@app.route('/emotion_history')
def get_emotion_history():
    with emotion_history_lock:
        return jsonify(emotion_history)


# Route to generate frames for live emotion detection
def generate_frames():
    global live_stats
    while camera_manager.is_running:
        frame = camera_manager.read_frame()
        if frame is None:
            continue

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            with live_stats_lock:
                live_stats['faces_count'] = len(faces)

            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)

                roi_gray = gray[y:y + h, x:x + w]
                if roi_gray.size == 0:
                    continue

                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                predictions = model.predict(roi_gray, verbose=0)
                emotion_index = np.argmax(predictions)
                emotion_label = emotion_labels[emotion_index]
                confidence = float(np.max(predictions) * 100)

                # Update live stats
                scores = {emotion_labels[i]: float(predictions[0][i] * 100) for i in range(len(emotion_labels))}
                with live_stats_lock:
                    live_stats['current_emotion'] = emotion_label
                    live_stats['confidence'] = round(confidence, 1)
                    live_stats['all_scores'] = {k: round(v, 1) for k, v in scores.items()}

                # Draw styled rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (138, 43, 226), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (138, 43, 226), -1)
                text = f"{emotion_label} {confidence:.0f}%"
                cv2.putText(frame, text, (x + 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Frame processing error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')