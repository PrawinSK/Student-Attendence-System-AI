import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
YUNET_PATH = os.path.join(MODELS_DIR, 'face_detection_yunet.onnx')
SFACE_PATH = os.path.join(MODELS_DIR, 'face_recognition_sface.onnx')

# Initialize Face Detector (YuNet CNN)
detector = cv2.FaceDetectorYN.create(
    model=YUNET_PATH,
    config='',
    input_size=(320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

# Initialize Face Recognizer (SFace 128-D Deep CNN embedding)
recognizer = cv2.FaceRecognizerSF.create(
    model=SFACE_PATH,
    config=''
)

def detect_faces(image_bgr):
    """Detects faces in an image using YuNet CNN."""
    h, w, _ = image_bgr.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(image_bgr)
    return faces if faces is not None else []

def extract_features(image_bgr, face):
    """Aligns and extracts 128-D feature encoding vector."""
    aligned_face = recognizer.alignCrop(image_bgr, face)
    features = recognizer.feature(aligned_face)
    return features.flatten() # 128-dimensional embedding vector

def compute_similarity(feat1, feat2):
    """
    Computes Euclidean (L2) distance and Cosine similarity.
    For SFace, L2 distance < 1.128 indicates the same identity.
    Cosine similarity > 0.363 indicates match.
    """
    f1 = feat1.reshape(1, -1)
    f2 = feat2.reshape(1, -1)
    l2_dist = recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_NORM_L2)
    cosine_sim = recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE)
    return l2_dist, cosine_sim

def encode_student_image(image_path):
    """Processes reference image and returns 128-D embedding."""
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = detect_faces(img)
    if len(faces) == 0:
        return None
    # Pick the largest face in reference photo
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    return extract_features(img, largest_face)

def crop_face(image_bgr, face, margin_ratio=0.25):
    """Crops face with contextual margin from image for clean display and re-training."""
    h, w, _ = image_bgr.shape
    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
    mw = int(fw * margin_ratio)
    mh = int(fh * margin_ratio)
    x1 = max(0, fx - mw)
    y1 = max(0, fy - mh)
    x2 = min(w, fx + fw + mw)
    y2 = min(h, fy + fh + mh)
    cropped = image_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return image_bgr[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    return cropped

