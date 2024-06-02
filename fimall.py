import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import numpy as np
import dlib
from imutils import face_utils
import tensorflow as tf
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تهيئة مكتشف الوجه ومحاكي العلامات من dlib
detector = dlib.get_frontal_face_detector()
predictor_path = ('\\Users\\Moath\\Downloads\\final project\\shape_predictor_68_face_landmarks.dat')
model_path = ('\\Users\\Moath\\Downloads\\CLASS_GUARD\\Emotions.h5')

try:
    predictor = dlib.shape_predictor(predictor_path)
except Exception as e:
    st.error(f"Failed to load shape predictor: {e}")
    logger.error(f"Failed to load shape predictor: {e}")

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    logger.error(f"Failed to load model: {e}")

# دالة لحساب المسافة الإقليدية
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# دالة للتحقق مما إذا كان العين مغمضة
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

# تعريف فئة VideoTransformer لـ Streamlit-webrtc
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.sleep = 0
        self.drowsy = 0
        self.active = 0
        self.status = ""
        self.color = (0, 0, 0)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        tf_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                self.sleep += 1
                self.drowsy = 0
                self.active = 0
                if self.sleep > 6:
                    self.status = "SLEEPING !!!"
                    self.color = (255, 0, 0)
            elif left_blink == 1 or right_blink == 1:
                self.sleep = 0
                self.active = 0
                self.drowsy += 1
                if self.drowsy > 6:
                    self.status = "Drowsy !"
                    self.color = (0, 0, 255)
            else:
                self.drowsy = 0
                self.sleep = 0
                self.active += 1
                if self.active > 6:
                    self.status = "Active "
                    self.color = (0, 255, 0)

            for (x, y, w, h) in tf_faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)
                predictions = model.predict(roi)[0]
                face_label = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise'][predictions.argmax()]
                face_label_position = (x, y - 10)
                cv2.putText(img, face_label, face_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(img, self.status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color, 3)

        return img

# واجهة المستخدم في Streamlit
st.title("Real-time Emotion and Drowsiness Detection")

# إضافة واجهة الدردشة
st.sidebar.title("Chat")
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

chat_input = st.sidebar.text_input("Enter your message:")
if st.sidebar.button("Send"):
    if chat_input:
        st.session_state['messages'].append(chat_input)
        chat_input = ""

st.sidebar.subheader("Chat history")
for msg in st.session_state['messages']:
    st.sidebar.text(msg)

# بث فيديو المسؤول
st.header("Admin Screen")
webrtc_streamer(key="admin_screen", video_transformer_factory=None, async_transform=True)

# بث فيديو المستخدمين
st.header("User Screens")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Screen 1")
    webrtc_streamer(key="user_screen_1", video_transformer_factory=VideoTransformer, async_transform=True)

with col2:
    st.subheader("Screen 2")
    webrtc_streamer(key="user_screen_2", video_transformer_factory=VideoTransformer, async_transform=True)
