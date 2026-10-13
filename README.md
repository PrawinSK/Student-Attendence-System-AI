# AI-Based Smart Attendance System

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/PrawinSK/Student-Attendence-System-AI)

An intelligent, computer-vision-powered attendance tracking platform built with **OpenCV (YuNet + SFace)**, **Flask**, and **SQLite**. The system performs fast and accurate facial detection and recognition on group classroom images, automatically logs unknown persons for verification, supports instant model training from newly identified photos, and exports period attendance reports directly to styled Excel spreadsheets.

---

## 🌟 Key Features

* **Real-Time Facial Detection & Recognition**:
  * Utilizes OpenCV's **YuNet Deep CNN** for robust multi-face detection in varying classroom lighting and orientations.
  * Employs **SFace Deep CNN** to compute 128-dimensional facial embedding vectors.
  * Multi-metric matching (Euclidean $L_2$ distance $< 1.128$ and Cosine similarity $> 0.363$).

* **Visual Facial Detection Overlay**:
  * **Green bounding boxes** indicate recognized enrolled students with their name and register number.
  * **Red bounding boxes** highlight unknown individuals with incident indices.

* **Unknown Person Identification & Review**:
  * Automatically crops faces of unidentified individuals from group images.
  * Both **Staff** (on the Classroom Portal) and the **HOD** (on the Analytics Dashboard) can:
    1. **Match Existing Student**: Link the cropped photo to an enrolled student, automatically caching their facial embedding for future matches.
    2. **Register as New Student**: Enroll the individual with their details (Name, Roll No, Department, Year).
    3. **Flag as Stranger / Visitor**: Log the unknown individual as an outsider in security incident records.

* **Continuous Model Auto-Training ("Photos Auto-Train")**:
  * New facial encodings are stored directly in the `student_photos` database table.
  * Future attendance sessions compare against both primary and newly learned embeddings, improving recognition accuracy without running manual retraining scripts.

* **Instant Period Attendance Excel Export**:
  * One-click download of styled `.xlsx` spreadsheets for any class period.
  * Automated dynamic filenames with Class Name, Subject, Period, Date, and Time (e.g., `Attendance_IT_Year4_DataStructures_Period3_2026-10-13_16-04-08.xlsx`).
  * Formatted with institution headers, attendance rates, status color highlights, and auto-adjusted column widths.

* **Student Lifecycle & Cascade Deletion**:
  * HOD can delete students with one click from the analytics console, automatically purging associated photos, facial embeddings, and attendance records.

---

## 🛠 Tech Stack

* **Backend**: Python 3, Flask, SQLite3, OpenCV (`cv2`), NumPy, openpyxl
* **Deep Learning Models**:
  * Face Detection: [YuNet ONNX](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  * Feature Extraction: [SFace ONNX](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
* **Frontend**: HTML5, Vanilla JavaScript, CSS3 (Modern Dark Theme)

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/PrawinSK/Student-Attendence-System-AI.git
cd Student-Attendence-System-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Models
The pre-trained ONNX models are included in the `models/` directory. If you ever need to redownload them:
```bash
python download_models.py
```

### 4. Run the Server
```bash
python app.py
```
The server will start at: **http://127.0.0.1:5000**

---

## 🔐 Default Credentials

| Role | Email | Password |
| :--- | :--- | :--- |
| **HOD Admin** | `hod@college.edu` | `hod123` |
| **Staff / Teacher** | `teacher@college.edu` | `password123` |

---

## 🧭 Navigation & Routes

| URL Route | Purpose |
| :--- | :--- |
| `/` | **Classroom Attendance Portal** — Upload class photos, view AI overlays, resolve unknown faces, and download period Excel reports. |
| `/hod_dashboard` | **HOD Analytics Console** — Department rosters, demographics, unknown incident review, and period attendance export. |
| `/add_student` | **Student Registration** — Enroll student profiles with reference portrait photos. |
| `/api/download_attendance_excel` | **Excel Export API** — Generates formatted `.xlsx` period sheets with class name, date, and time. |
| `/api/resolve_unknown_face` | **Face Resolution API** — Resolves unknown detections (match existing, add new student, or mark stranger). |
| `/api/delete_student` | **Student Deletion API** — Cascade deletes student profile, facial encodings, and attendance records. |

---

## 📄 License
This project is licensed under the MIT License.
