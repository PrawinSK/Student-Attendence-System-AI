import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFace_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

yunet_path = os.path.join(MODELS_DIR, 'face_detection_yunet.onnx')
sface_path = os.path.join(MODELS_DIR, 'face_recognition_sface.onnx')

def download_models():
    if not os.path.exists(yunet_path) or os.path.getsize(yunet_path) == 0:
        print("Downloading YuNet Face Detection model...")
        urllib.request.urlretrieve(YUNET_URL, yunet_path)
        print("YuNet downloaded successfully.")
    else:
        print("YuNet model exists.")

    if not os.path.exists(sface_path) or os.path.getsize(sface_path) == 0:
        print("Downloading SFace 128-D Feature Extraction model...")
        urllib.request.urlretrieve(SFace_URL, sface_path)
        print("SFace downloaded successfully.")
    else:
        print("SFace model exists.")

if __name__ == '__main__':
    download_models()
