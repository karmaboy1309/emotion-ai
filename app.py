from flask import Flask, render_template, request, Response, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import cv2
from mtcnn import MTCNN
import threading
import time
from datetime import datetime
import base64
from werkzeug.utils import secure_filename

# Flask App Initialize
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB Max Upload Limit

# Allowed File Extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'avif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure upload directory exists
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Thread Lock for Model & Detector Thread-Safety
model_lock = threading.Lock()

# Model Load
model = load_model("models/emotion_model.keras")

# Compiled tf.function warm-up for fast low-latency CPU inference
@tf.function(reduce_retracing=True)
def _fast_model_predict(batch_tensor):
    return model(batch_tensor, training=False)

try:
    _dummy_input = tf.zeros((1, 48, 48, 1), dtype=tf.float32)
    _ = _fast_model_predict(_dummy_input)
except Exception as _e:
    print(f"Model warm-up status: {_e}")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion emoji mapping
emotion_emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

# Initialize MTCNN for accurate face detection (used for still images)
detector = MTCNN()

# Lightweight detector for live stream (faster than MTCNN)
haar_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Live stream tuning
LIVE_TARGET_FPS = 10
LIVE_DETECT_EVERY_N = 4
LIVE_MAX_WIDTH = 320
LIVE_JPEG_QUALITY = 60

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


def cleanup_old_uploads(max_age_seconds=600):
    """Clean up uploaded files older than max_age_seconds (default 10 minutes)"""
    try:
        now = time.time()
        for f in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(file_path):
                if now - os.path.getmtime(file_path) > max_age_seconds:
                    os.remove(file_path)
    except Exception as e:
        print(f"Cleanup error: {e}")


def predict_faces_batch(face_crops):
    """
    High-performance vectorized batch inference for face crop numpy arrays.
    Returns array of prediction probability distributions for all faces.
    Bypasses Keras model.predict() overhead for maximum throughput.
    """
    if not face_crops:
        return np.array([])

    batch = np.array(face_crops, dtype=np.float32)
    with model_lock:
        tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        predictions = _fast_model_predict(tensor).numpy()
    return predictions



# Camera manager for proper resource handling (Server-side stream fallback)
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
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                self.camera.set(cv2.CAP_PROP_FPS, 15)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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


# Security Header Response Middleware
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


# Route for Home Page
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        cleanup_old_uploads()
        file = request.files.get("image")
        if file and file.filename != '':
            if not allowed_file(file.filename):
                error = "Invalid file extension. Allowed formats: PNG, JPG, JPEG, WEBP, AVIF."
                return render_template("index.html", result=None, error=error)

            try:
                safe_name = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, safe_name)
                file.save(file_path)

                frame = cv2.imread(file_path)
                if frame is None:
                    error = "Error: Could not decode image file."
                    return render_template("index.html", result=None, error=error)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with model_lock:
                    faces = detector.detect_faces(rgb_frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if len(faces) == 0:
                    error = "No face detected in the image. Please upload a clear photo with a visible face."
                    return render_template("index.html", result=None, error=error)

                face_crops = []
                valid_boxes = []

                for face in faces:
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)

                    sub_face_img = gray[y:y + h, x:x + w]
                    if sub_face_img.size == 0:
                        continue

                    resized = cv2.resize(sub_face_img, (48, 48))
                    normalize = resized / 255.0
                    reshaped = np.expand_dims(normalize, axis=-1)
                    face_crops.append(reshaped)
                    valid_boxes.append((x, y, w, h))

                if not face_crops:
                    error = "Could not extract valid face regions from the image."
                    return render_template("index.html", result=None, error=error)

                batch_predictions = predict_faces_batch(face_crops)
                all_faces_data = []

                for idx, (x, y, w, h) in enumerate(valid_boxes):
                    result = batch_predictions[idx]
                    label = int(np.argmax(result))
                    confidence = float(np.max(result) * 100)

                    scores = {emotion_labels[i]: float(result[i] * 100) for i in range(len(emotion_labels))}

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

                output_filename = "output_" + safe_name
                output_image_path = os.path.join(UPLOAD_FOLDER, output_filename)
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
                                       image=output_filename,
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


# Client-Side REST API Endpoint for Live Camera Stream Emotion Detection
@app.route("/api/detect_emotion", methods=["POST"])
def api_detect_emotion():
    """
    Asynchronous client-side REST API for real-time live webcam emotion detection.
    Accepts Base64 image payload from browser webcam canvas.
    Returns detected faces, bounding boxes, predictions, confidence distributions, and inference latency.
    """
    t_start = time.perf_counter()
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "Missing image payload", "faces": [], "inference_ms": 0}), 400

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image frame", "faces": [], "inference_ms": 0}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = haar_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        inference_ms = round((time.perf_counter() - t_start) * 1000, 1)

        if len(faces) == 0:
            return jsonify({"faces_count": 0, "faces": [], "inference_ms": inference_ms}), 200

        face_crops = []
        valid_coords = []

        for (x, y, w, h) in faces:
            x, y = max(0, int(x)), max(0, int(y))
            w, h = int(w), int(h)
            roi_gray = gray[y:y + h, x:x + w]
            if roi_gray.size == 0:
                continue

            resized = cv2.resize(roi_gray, (48, 48)) / 255.0
            reshaped = np.expand_dims(resized, axis=-1)
            face_crops.append(reshaped)
            valid_coords.append((x, y, w, h))

        if not face_crops:
            inference_ms = round((time.perf_counter() - t_start) * 1000, 1)
            return jsonify({"faces_count": 0, "faces": [], "inference_ms": inference_ms}), 200

        batch_predictions = predict_faces_batch(face_crops)
        faces_data = []

        for idx, (x, y, w, h) in enumerate(valid_coords):
            probs = batch_predictions[idx]
            label_idx = int(np.argmax(probs))
            label = emotion_labels[label_idx]
            confidence = float(np.max(probs) * 100)
            scores = {emotion_labels[i]: round(float(probs[i] * 100), 1) for i in range(len(emotion_labels))}

            faces_data.append({
                "box": [x, y, w, h],
                "emotion": label,
                "emoji": emotion_emojis[label],
                "confidence": round(confidence, 1),
                "scores": scores
            })

        inference_ms = round((time.perf_counter() - t_start) * 1000, 1)
        return jsonify({
            "faces_count": len(faces_data),
            "faces": faces_data,
            "inference_ms": inference_ms
        }), 200

    except Exception as e:
        inference_ms = round((time.perf_counter() - t_start) * 1000, 1)
        return jsonify({"error": str(e), "faces": [], "inference_ms": inference_ms}), 500


# Route to start the camera (Server-side fallback)
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


# Route to generate frames for live emotion detection (Server-side fallback stream)
def generate_frames():
    global live_stats
    last_faces = []
    frame_index = 0
    last_emit = 0.0
    while camera_manager.is_running:
        frame = camera_manager.read_frame()
        if frame is None:
            continue

        try:
            frame_index += 1
            do_detect = (frame_index % LIVE_DETECT_EVERY_N) == 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if do_detect:
                height, width = frame.shape[:2]
                scale = 1.0
                small_frame = frame
                if width > LIVE_MAX_WIDTH:
                    scale = width / LIVE_MAX_WIDTH
                    small_frame = cv2.resize(frame, (LIVE_MAX_WIDTH, int(height / scale)))

                gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = haar_detector.detectMultiScale(
                    gray_small,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40)
                )

                face_crops = []
                face_coords = []

                for (x, y, w, h) in faces:
                    x, y = max(0, x), max(0, y)
                    x_full = int(x * scale)
                    y_full = int(y * scale)
                    w_full = int(w * scale)
                    h_full = int(h * scale)

                    roi_gray = gray[y_full:y_full + h_full, x_full:x_full + w_full]
                    if roi_gray.size == 0:
                        continue

                    roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
                    roi_gray = np.expand_dims(roi_gray, axis=-1)
                    face_crops.append(roi_gray)
                    face_coords.append((x_full, y_full, w_full, h_full))

                last_faces = []
                best_face = None
                best_conf = -1.0

                if face_crops:
                    batch_predictions = predict_faces_batch(face_crops)
                    for idx, (x, y, w, h) in enumerate(face_coords):
                        predictions = batch_predictions[idx]
                        emotion_index = int(np.argmax(predictions))
                        emotion_label = emotion_labels[emotion_index]
                        confidence = float(np.max(predictions) * 100)
                        scores = {emotion_labels[i]: float(predictions[i] * 100) for i in range(len(emotion_labels))}

                        face_data = {
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'emotion': emotion_label,
                            'confidence': confidence,
                            'scores': scores
                        }
                        last_faces.append(face_data)

                        if confidence > best_conf:
                            best_conf = confidence
                            best_face = face_data

                with live_stats_lock:
                    live_stats['faces_count'] = len(last_faces)
                    if best_face:
                        live_stats['current_emotion'] = best_face['emotion']
                        live_stats['confidence'] = round(best_face['confidence'], 1)
                        live_stats['all_scores'] = {k: round(v, 1) for k, v in best_face['scores'].items()}
                    else:
                        live_stats['current_emotion'] = None
                        live_stats['confidence'] = 0
                        live_stats['all_scores'] = {}
            else:
                with live_stats_lock:
                    live_stats['faces_count'] = len(last_faces)

            for face in last_faces:
                x, y, w, h = face['x'], face['y'], face['w'], face['h']
                emotion_label = face['emotion']
                confidence = face['confidence']

                # Draw styled rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (138, 43, 226), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (138, 43, 226), -1)
                text = f"{emotion_label} {confidence:.0f}%"
                cv2.putText(frame, text, (x + 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Frame processing error: {e}")

        now = time.time()
        if now - last_emit < (1.0 / LIVE_TARGET_FPS):
            time.sleep(0.002)
            continue
        last_emit = now

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_JPEG_QUALITY])
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