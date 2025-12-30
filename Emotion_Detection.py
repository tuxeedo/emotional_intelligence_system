import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===========================
# LOAD EMOTION MODEL
# ===========================
MODEL_PATH = r"ML_MODEL/ML_Model/emotion_detection_model.h5"

try:
    model = load_model(MODEL_PATH)
    print("✅ Emotion model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)
    exit()

# ===========================
# EMOTION LABELS
# ===========================
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# ===========================
# LOAD FACE CASCADE
# ===========================
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("❌ Haar Cascade not loaded")
    exit()
else:
    print("✅ Haar Cascade loaded")

# ===========================
# PREPROCESS FACE
# ===========================
def preprocess_face(face_img):
    """
    Convert face ROI to model input format
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype("float32") / 255.0
    reshaped = normalized.reshape(1, 48, 48, 1)
    return reshaped

# ===========================
# OPEN WEBCAM (WINDOWS FIX)
# ===========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()
else:
    print("✅ Webcam opened successfully")

# ===========================
# MAIN LOOP
# ===========================
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Loop through detected faces
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        try:
            processed_face = preprocess_face(face_roi)
            predictions = model.predict(processed_face, verbose=0)

            emotion_index = np.argmax(predictions)
            emotion_text = emotion_labels[emotion_index]
            confidence = np.max(predictions) * 100

        except Exception as e:
            print("Prediction error:", e)
            continue

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display emotion
        label = f"{emotion_text} ({confidence:.1f}%)"
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # Show output
    cv2.imshow("Emotion Detection System", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===========================
# CLEANUP
# ===========================
cap.release()
cv2.destroyAllWindows()
print("🛑 Program terminated")
