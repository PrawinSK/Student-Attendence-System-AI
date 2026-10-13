import os
import datetime
import uuid
import shutil
import io
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
import cv2
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from database import init_db, get_db_connection
from face_engine import detect_faces, extract_features, compute_similarity, encode_student_image, crop_face

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
UNKNOWN_FOLDER = os.path.join(UPLOAD_FOLDER, 'unknown_faces')
ANNOTATED_FOLDER = os.path.join(UPLOAD_FOLDER, 'annotated')

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

def _save_student_image(file, register_number):
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    filename = f"{register_number}{ext}"
    filepath = os.path.join(DATASET_DIR, filename)
    file.save(filepath)
    return os.path.join('dataset', filename).replace('\\', '/')

def _require_hod(request_data):
    teacher_id = request_data.get('hod_id')
    if not teacher_id:
        return None
    conn = get_db_connection()
    t = conn.execute("SELECT * FROM teachers WHERE id = ? AND role = 'hod'", (teacher_id,)).fetchone()
    conn.close()
    return t

def process_attendance_image(filepath, department, year, subject="General", period=1, date_str=None):
    """
    Core AI logic: detects faces in group classroom image,
    computes 128-D embeddings, and matches against students enrolled in (dept, year).
    Crops unknown faces, draws color-coded bounding boxes (green for enrolled, red for unknown),
    and logs unknown individuals to DB for Staff and HOD verification & auto-training.
    """
    if date_str is None:
        date_str = datetime.date.today().isoformat()

    img = cv2.imread(filepath)
    if img is None:
        return [], 0, [], None

    detected_faces = detect_faces(img)

    conn = get_db_connection()
    students = conn.execute("SELECT * FROM students WHERE department = ? AND year = ?", (department, year)).fetchall()

    # Pre-compute or load all reference face vectors for this cohort (primary + extra learned photos)
    enrolled_embeddings = {}
    for st in students:
        embs = []
        # Primary photo
        full_img_path = os.path.join(BASE_DIR, st['image_path'])
        if os.path.exists(full_img_path):
            emb = encode_student_image(full_img_path)
            if emb is not None:
                embs.append(emb)

        # Extra photos / auto-trained embeddings from student_photos table
        extras = conn.execute("SELECT image_path, embedding FROM student_photos WHERE student_id = ?", (st['id'],)).fetchall()
        for ep in extras:
            if ep['embedding']:
                embs.append(np.frombuffer(ep['embedding'], dtype=np.float32))
            elif ep['image_path']:
                ep_full = os.path.join(BASE_DIR, ep['image_path'])
                if os.path.exists(ep_full):
                    e = encode_student_image(ep_full)
                    if e is not None:
                        embs.append(e)

        if embs:
            enrolled_embeddings[st['id']] = (st, embs)

    session_id = str(uuid.uuid4())[:8]
    annotated_img = img.copy()
    img_h, img_w = img.shape[:2]

    recognized_ids = set()
    unknown_faces_detected = []
    unk_counter = 0

    for face in detected_faces:
        face_feature = extract_features(img, face)
        matched_id = None
        min_dist = float('inf')

        for s_id, (st_data, ref_embs) in enrolled_embeddings.items():
            for ref_emb in ref_embs:
                l2_dist, cosine_sim = compute_similarity(face_feature, ref_emb)
                # Match thresholds: L2 distance < 1.128 or Cosine > 0.363
                if (l2_dist < 1.128 or cosine_sim > 0.363) and l2_dist < min_dist:
                    min_dist = l2_dist
                    matched_id = s_id

        fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        x1, y1 = max(0, fx), max(0, fy)
        x2, y2 = min(img_w, fx + fw), min(img_h, fy + fh)

        if matched_id is not None:
            recognized_ids.add(matched_id)
            st_info = enrolled_embeddings[matched_id][0]
            label = f"{st_info['name']} ({st_info['register_number']})"
            # Green bounding box for recognized students
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (46, 204, 113), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, max(0, y1 - 22)), (min(img_w, x1 + tw + 8), y1), (46, 204, 113), -1)
            cv2.putText(annotated_img, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            unk_counter += 1
            label = f"Unknown #{unk_counter}"
            # Red bounding box for unknown individuals
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 240), 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated_img, (x1, max(0, y1 - 24)), (min(img_w, x1 + tw + 8), y1), (0, 0, 240), -1)
            cv2.putText(annotated_img, label, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Crop face cleanly with contextual margin
            crop_bgr = crop_face(img, face)
            crop_filename = f"unk_{session_id}_{unk_counter}.jpg"
            crop_path = os.path.join(UNKNOWN_FOLDER, crop_filename)
            cv2.imwrite(crop_path, crop_bgr)
            crop_rel = f"uploads/unknown_faces/{crop_filename}"

            unknown_faces_detected.append({
                'index': unk_counter,
                'crop_rel': crop_rel,
                'crop_url': f"/{crop_rel}",
                'label': label,
                'embedding_bytes': face_feature.astype(np.float32).tobytes(),
                'box': [fx, fy, fw, fh]
            })

    # Save annotated classroom photo
    annotated_filename = f"annotated_{session_id}.jpg"
    annotated_path = os.path.join(ANNOTATED_FOLDER, annotated_filename)
    cv2.imwrite(annotated_path, annotated_img)
    annotated_rel = f"uploads/annotated/{annotated_filename}"

    orig_rel = filepath.replace(BASE_DIR, '').lstrip('\\/').replace('\\', '/')

    # Save unknown detections to DB for Staff and HOD review
    c = conn.cursor()
    saved_unknown_faces = []
    for unk in unknown_faces_detected:
        c.execute('''
            INSERT INTO unknown_faces
            (session_id, department, year, subject, period, date, crop_image_path, original_image_path, annotated_image_path, embedding, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        ''', (
            session_id, department, int(year), subject, int(period), date_str,
            unk['crop_rel'], orig_rel, annotated_rel, unk['embedding_bytes']
        ))
        unk_id = c.lastrowid
        saved_unknown_faces.append({
            'id': unk_id,
            'session_id': session_id,
            'crop_url': unk['crop_url'],
            'label': unk['label'],
            'status': 'pending',
            'department': department,
            'year': year,
            'subject': subject,
            'period': period
        })

    conn.commit()
    conn.close()

    # Build student response list
    students_list = []
    for st in students:
        status = 'Present' if st['id'] in recognized_ids else 'Absent'
        students_list.append({
            'id': st['id'],
            'name': st['name'],
            'register_number': st['register_number'],
            'department': st['department'],
            'year': st['year'],
            'status': status
        })

    return students_list, unk_counter, saved_unknown_faces, f"/{annotated_rel}"

# --- REST & WEB ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/add_student', methods=['GET', 'POST'])
def add_student_view():
    if request.method == 'GET':
        return render_template('add_student.html')
    if 'image' not in request.files:
        return render_template('add_student.html', error='No image provided')
    file = request.files['image']
    name = request.form.get('name')
    register_number = request.form.get('register_number')
    department = request.form.get('department')
    year = request.form.get('year')
    if file.filename == '':
        return render_template('add_student.html', error='No image selected')
    if not all([name, register_number, department, year]):
        return render_template('add_student.html', error='Missing required fields')
    
    db_image_path = _save_student_image(file, register_number)
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO students (name, register_number, department, year, image_path) VALUES (?, ?, ?, ?, ?)',
                  (name, register_number, department, int(year), db_image_path))
        new_id = c.lastrowid
        # Pre-compute and store embedding for auto-training
        full_img = os.path.join(BASE_DIR, db_image_path)
        emb = encode_student_image(full_img)
        if emb is not None:
            c.execute('INSERT INTO student_photos (student_id, image_path, embedding) VALUES (?, ?, ?)',
                      (new_id, db_image_path, emb.astype(np.float32).tobytes()))
        conn.commit()
        return render_template('add_student.html', success=f'Successfully added {name} ({register_number})!')
    except Exception:
        conn.rollback()
        return render_template('add_student.html', error='Register number already exists or DB error.')
    finally:
        conn.close()

@app.route('/api/add_student', methods=['POST'])
def api_add_student():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
    file = request.files['image']
    name = request.form.get('name')
    register_number = request.form.get('register_number')
    department = request.form.get('department')
    year = request.form.get('year')
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty file'}), 400
    if not all([name, register_number, department, year]):
        return jsonify({'status': 'error', 'message': 'Missing fields'}), 400
    db_image_path = _save_student_image(file, register_number)
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO students (name, register_number, department, year, image_path) VALUES (?, ?, ?, ?, ?)',
                  (name, register_number, department, int(year), db_image_path))
        new_id = c.lastrowid
        full_img = os.path.join(BASE_DIR, db_image_path)
        emb = encode_student_image(full_img)
        if emb is not None:
            c.execute('INSERT INTO student_photos (student_id, image_path, embedding) VALUES (?, ?, ?)',
                      (new_id, db_image_path, emb.astype(np.float32).tobytes()))
        conn.commit()
        return jsonify({'status': 'success', 'message': f'Successfully added {name}!'})
    except Exception:
        conn.rollback()
        return jsonify({'status': 'error', 'message': 'Register number already exists or DB error.'}), 400
    finally:
        conn.close()

@app.route('/api/delete_student', methods=['POST', 'DELETE'])
def api_delete_student():
    """Permanently deletes a student, their facial encodings, attendance history, and dataset photos."""
    data = request.json or request.form
    student_id = data.get('student_id')
    if not student_id:
        return jsonify({'status': 'error', 'message': 'student_id required'}), 400

    conn = get_db_connection()
    c = conn.cursor()
    student = c.execute('SELECT * FROM students WHERE id = ?', (student_id,)).fetchone()
    if not student:
        conn.close()
        return jsonify({'status': 'error', 'message': 'Student not found'}), 404

    try:
        # 1. Delete associated student_photos records and files
        photos = c.execute('SELECT image_path FROM student_photos WHERE student_id = ?', (student_id,)).fetchall()
        for p in photos:
            p_full = os.path.join(BASE_DIR, p['image_path'])
            if os.path.exists(p_full):
                try:
                    os.remove(p_full)
                except Exception:
                    pass
        c.execute('DELETE FROM student_photos WHERE student_id = ?', (student_id,))

        # 2. Delete attendance records for this student
        c.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))

        # 3. Unlink from unknown_faces if any
        c.execute('UPDATE unknown_faces SET resolved_student_id = NULL WHERE resolved_student_id = ?', (student_id,))

        # 4. Remove primary image file from dataset if exists
        primary_full = os.path.join(BASE_DIR, student['image_path'])
        if os.path.exists(primary_full):
            try:
                os.remove(primary_full)
            except Exception:
                pass

        # 5. Delete student record
        c.execute('DELETE FROM students WHERE id = ?', (student_id,))
        conn.commit()

        return jsonify({
            'status': 'success',
            'message': f"Student {student['name']} ({student['register_number']}) was successfully deleted."
        })
    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.json or request.form
    email = data.get('email')
    password = data.get('password')
    conn = get_db_connection()
    teacher = conn.execute('SELECT * FROM teachers WHERE email = ? AND password = ?', (email, password)).fetchone()
    conn.close()
    if teacher:
        return jsonify({
            'status': 'success',
            'teacher_id': teacher['id'],
            'name': teacher['name'],
            'role': teacher['role']
        })
    return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

@app.route('/hod/create_teacher', methods=['POST'])
def hod_create_teacher():
    data = request.json or request.form
    if not _require_hod(data):
        return jsonify({'status': 'error', 'message': 'Unauthorized – HOD only'}), 403
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    department = data.get('department')
    year = data.get('year')
    subject = data.get('subject')
    periods = data.get('periods', [])
    if isinstance(periods, str):
        periods = [int(p.strip()) for p in periods.split(',') if p.strip()]

    if not all([name, email, password]):
        return jsonify({'status': 'error', 'message': 'Missing fields'}), 400
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute('INSERT INTO teachers (name, email, password, role) VALUES (?, ?, ?, ?)', (name, email, password, 'teacher'))
        teacher_id = c.lastrowid
        if department and year and subject and periods:
            for p in periods:
                c.execute('''INSERT OR IGNORE INTO teacher_assignments
                             (teacher_id, department, year, subject, period)
                             VALUES (?, ?, ?, ?, ?)''', (teacher_id, department, int(year), subject, int(p)))
        conn.commit()
        return jsonify({'status': 'success', 'message': 'Teacher created and assigned!'})
    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

@app.route('/teacher/dashboard', methods=['GET'])
def teacher_dashboard():
    teacher_id = request.args.get('teacher_id')
    if not teacher_id:
        return jsonify({'status': 'error', 'message': 'Missing teacher_id'}), 400
    conn = get_db_connection()
    assignments = conn.execute('SELECT * FROM teacher_assignments WHERE teacher_id = ?', (teacher_id,)).fetchall()
    conn.close()
    if not assignments:
        return jsonify({'status': 'success', 'assignments': [], 'message': 'No assignments yet – contact HOD'})

    grouped = {}
    for a in assignments:
        key = f"{a['department']}|{a['year']}|{a['subject']}"
        if key not in grouped:
            grouped[key] = {
                'department': a['department'],
                'year': a['year'],
                'subject': a['subject'],
                'periods': []
            }
        grouped[key]['periods'].append(a['period'])

    return jsonify({'status': 'success', 'assignments': list(grouped.values())})

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
    file = request.files['image']
    teacher_id = request.form.get('teacher_id')
    department = request.form.get('department')
    year = int(request.form.get('year', 0))
    subject = request.form.get('subject') or "General"
    period = int(request.form.get('period', 1))

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty file'}), 400

    if teacher_id:
        conn = get_db_connection()
        teacher = conn.execute("SELECT role FROM teachers WHERE id = ?", (teacher_id,)).fetchone()
        is_hod = teacher and teacher['role'] == 'hod'
        if not is_hod:
            assignment = conn.execute('''
                SELECT id FROM teacher_assignments
                WHERE teacher_id = ? AND department = ? AND year = ? AND subject = ? AND period = ?
            ''', (teacher_id, department, year, subject, period)).fetchone()
            conn.close()
            if not assignment:
                return jsonify({'status': 'error', 'message': 'Unauthorized access: You are not assigned to this class/period.'}), 403
        else:
            conn.close()

    filepath = os.path.join(UPLOAD_FOLDER, f"temp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
    file.save(filepath)

    try:
        students_list, unknown_count, unknown_faces_detected, annotated_image_url = process_attendance_image(
            filepath, department, year, subject=subject, period=period
        )
        return jsonify({
            'status': 'success',
            'students': students_list,
            'unknown_count': unknown_count,
            'unknown_faces': unknown_faces_detected,
            'annotated_image': annotated_image_url
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/save_attendance', methods=['POST'])
def save_attendance():
    data = request.json
    students = data.get('students', [])
    subject = data.get('subject')
    period = int(data.get('period', 0))
    date_str = data.get('date', datetime.date.today().isoformat())

    conn = get_db_connection()
    c = conn.cursor()
    try:
        seen_ids = set()
        for st in students:
            s_id = st.get('id')
            if not s_id or s_id in seen_ids:
                continue
            seen_ids.add(s_id)
            existing = c.execute('''
                SELECT id FROM attendance
                WHERE student_id = ? AND date = ? AND period = ? AND subject = ?
            ''', (s_id, date_str, period, subject)).fetchone()
            if existing:
                c.execute('UPDATE attendance SET status = ? WHERE id = ?', (st['status'], existing['id']))
            else:
                c.execute('''
                    INSERT INTO attendance (student_id, date, period, subject, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (s_id, date_str, period, subject, st['status']))
        conn.commit()
        return jsonify({'status': 'success', 'message': 'Attendance saved successfully!'})
    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

def generate_attendance_excel(department, year, subject, period, date_str, time_str, students_list):
    """
    Builds a professional, styled Excel (.xlsx) spreadsheet for period attendance,
    including metadata banner, class cohort info, summary statistics, and styled student roster.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"{department} Y{year} P{period}"
    ws.views.sheetView[0].showGridLines = True

    # Styling fonts & fills
    font_title = Font(name="Calibri", size=15, bold=True, color="FFFFFF")
    fill_title = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")

    font_meta_lbl = Font(name="Calibri", size=10, bold=True, color="64748B")
    font_meta_val = Font(name="Calibri", size=10, bold=True, color="0F172A")

    font_tbl_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    fill_tbl_header = PatternFill(start_color="2563EB", end_color="2563EB", fill_type="solid")

    font_present = Font(name="Calibri", size=11, bold=True, color="065F46")
    fill_present = PatternFill(start_color="D1FAE5", end_color="D1FAE5", fill_type="solid")

    font_absent = Font(name="Calibri", size=11, bold=True, color="991B1B")
    fill_absent = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")

    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )

    # 1. Header Banner
    ws.merge_cells('A1:G1')
    c1 = ws['A1']
    c1.value = "SMART ATTENDANCE SYSTEM - PERIOD ATTENDANCE REPORT"
    c1.font = font_title
    c1.fill = fill_title
    c1.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 40

    # 2. Metadata Section
    ws.merge_cells('A2:B2'); ws['A2'] = "Class / Department:"; ws['A2'].font = font_meta_lbl
    ws['C2'] = f"{department} (Year {year})"; ws['C2'].font = font_meta_val
    ws['D2'] = "Subject:"; ws['D2'].font = font_meta_lbl
    ws.merge_cells('E2:G2'); ws['E2'] = str(subject); ws['E2'].font = font_meta_val

    ws.merge_cells('A3:B3'); ws['A3'] = "Period:"; ws['A3'].font = font_meta_lbl
    ws['C3'] = f"Period {period}"; ws['C3'].font = font_meta_val
    ws['D3'] = "Report Date & Time:"; ws['D3'].font = font_meta_lbl
    ws.merge_cells('E3:G3'); ws['E3'] = f"{date_str} {time_str}"; ws['E3'].font = font_meta_val

    present_count = sum(1 for s in students_list if s.get('status') == 'Present')
    absent_count = len(students_list) - present_count
    rate = round((present_count / len(students_list) * 100), 1) if students_list else 0

    ws.merge_cells('A4:G4')
    ws['A4'] = f"Attendance Summary:  Total Enrolled: {len(students_list)}  |  Present: {present_count}  |  Absent: {absent_count}  |  Attendance Rate: {rate}%"
    ws['A4'].font = Font(name="Calibri", size=10, bold=True, color="1E3A8A")
    ws['A4'].fill = PatternFill(start_color="DBEAFE", end_color="DBEAFE", fill_type="solid")
    ws['A4'].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[4].height = 24

    ws.append([]) # Row 5 spacer

    # 3. Table Column Headers
    headers = ["S.No", "Register Number", "Student Name", "Department", "Academic Year", "Attendance Status", "Verification Method"]
    ws.append(headers)
    ws.row_dimensions[6].height = 26
    for col_idx in range(1, 8):
        cell = ws.cell(row=6, column=col_idx)
        cell.font = font_tbl_header
        cell.fill = fill_tbl_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    # 4. Student Roster Records
    for idx, st in enumerate(students_list, start=1):
        status = st.get('status', 'Absent')
        row_data = [
            idx,
            st.get('register_number', ''),
            st.get('name', ''),
            st.get('department', department),
            f"Year {st.get('year', year)}",
            status,
            "AI Face Recognition (YuNet + SFace)"
        ]
        ws.append(row_data)
        curr_row = 6 + idx
        ws.row_dimensions[curr_row].height = 22

        for col_idx in range(1, 8):
            cell = ws.cell(row=curr_row, column=col_idx)
            cell.border = thin_border
            cell.alignment = Alignment(vertical="center", horizontal="center" if col_idx in [1, 4, 5, 6] else "left")

            if col_idx == 6:
                if status == 'Present':
                    cell.font = font_present
                    cell.fill = fill_present
                else:
                    cell.font = font_absent
                    cell.fill = fill_absent

    # Auto-adjust column widths
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 18
    ws.column_dimensions['C'].width = 24
    ws.column_dimensions['F'].width = 18
    ws.column_dimensions['G'].width = 30

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return out

@app.route('/api/download_attendance_excel', methods=['GET', 'POST'])
def api_download_attendance_excel():
    """
    Generates and downloads an Excel spreadsheet for a period with:
    Filename: Attendance_{Class}_{Subject}_Period{Period}_{Date}_{Time}.xlsx
    """
    data = (request.json or {}) if request.is_json else (request.form.to_dict() or request.args.to_dict())
    department = data.get('department', 'General')
    year = int(data.get('year', 1))
    subject = data.get('subject', 'General')
    period = int(data.get('period', 1))
    date_str = data.get('date') or datetime.date.today().isoformat()
    now_dt = datetime.datetime.now()
    time_str = now_dt.strftime("%H:%M:%S")

    students = data.get('students')
    if not students:
        conn = get_db_connection()
        # Group by student id to strictly prevent any duplicate rows
        records = conn.execute('''
            SELECT s.id, s.name, s.register_number, s.department, s.year,
                   COALESCE(MAX(a.status), 'Absent') as status
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id AND a.date = ? AND a.period = ? AND a.subject = ?
            WHERE s.department = ? AND s.year = ?
            GROUP BY s.id, s.name, s.register_number, s.department, s.year
            ORDER BY s.name
        ''', (date_str, period, subject, department, int(year))).fetchall()
        conn.close()
        students = [dict(r) for r in records]
    else:
        # Strictly deduplicate student list passed from client
        deduped = {}
        for st in students:
            key = str(st.get('register_number') or st.get('id'))
            if key not in deduped or st.get('status') == 'Present':
                deduped[key] = st
        students = list(deduped.values())

    clean_dept = str(department).replace(' ', '_')
    clean_subj = "".join(c for c in str(subject) if c.isalnum() or c in ('_', '-')) or "General"
    time_file_str = now_dt.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Attendance_{clean_dept}_Year{year}_{clean_subj}_Period{period}_{time_file_str}.xlsx"

    excel_stream = generate_attendance_excel(
        department=department,
        year=year,
        subject=subject,
        period=period,
        date_str=date_str,
        time_str=time_str,
        students_list=students
    )

    return send_file(
        excel_stream,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/check_student', methods=['GET'])
def api_check_student():
    """Verify if a student exists by register / roll number."""
    reg_no = request.args.get('register_number', '').strip()
    if not reg_no:
        return jsonify({'status': 'error', 'message': 'Register number required'}), 400
    conn = get_db_connection()
    student = conn.execute('SELECT * FROM students WHERE register_number = ?', (reg_no,)).fetchone()
    conn.close()
    if student:
        return jsonify({
            'status': 'success',
            'exists': True,
            'student': {
                'id': student['id'],
                'name': student['name'],
                'register_number': student['register_number'],
                'department': student['department'],
                'year': student['year'],
                'image_path': student['image_path']
            }
        })
    return jsonify({'status': 'success', 'exists': False})

@app.route('/api/resolve_unknown_face', methods=['POST'])
def api_resolve_unknown_face():
    """
    Resolves an unknown person detection for HOD and Staff:
    1. 'match_existing': Links cropped face to existing student, auto-trains model embedding, marks Present.
    2. 'add_new': Creates new student using the cropped face, auto-trains model embedding, marks Present.
    3. 'mark_stranger': Flags individual as outsider/stranger.
    """
    data = request.json or request.form
    unknown_id = data.get('unknown_id')
    action = data.get('action') # 'match_existing', 'add_new', 'mark_stranger'
    resolved_by = data.get('resolved_by', 'staff')

    if not unknown_id or not action:
        return jsonify({'status': 'error', 'message': 'Missing unknown_id or action'}), 400

    conn = get_db_connection()
    c = conn.cursor()
    unk = c.execute('SELECT * FROM unknown_faces WHERE id = ?', (unknown_id,)).fetchone()
    if not unk:
        conn.close()
        return jsonify({'status': 'error', 'message': 'Unknown face record not found'}), 404

    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    crop_full_path = os.path.join(BASE_DIR, unk['crop_image_path'])

    try:
        if action == 'match_existing':
            reg_no = data.get('register_number', '').strip()
            student_id = data.get('student_id')
            if reg_no:
                st = c.execute('SELECT * FROM students WHERE register_number = ?', (reg_no,)).fetchone()
            elif student_id:
                st = c.execute('SELECT * FROM students WHERE id = ?', (student_id,)).fetchone()
            else:
                conn.close()
                return jsonify({'status': 'error', 'message': 'Register number or student_id required'}), 400

            if not st:
                conn.close()
                return jsonify({'status': 'error', 'message': 'Student with that register number does not exist'}), 404

            # 1. Copy cropped face to dataset folder for student
            dataset_filename = f"{st['register_number']}_learned_{ts}.jpg"
            dataset_dest = os.path.join(DATASET_DIR, dataset_filename)
            if os.path.exists(crop_full_path):
                shutil.copy2(crop_full_path, dataset_dest)
            db_dataset_path = f"dataset/{dataset_filename}"

            # 2. Save embedding and photo into student_photos (auto-trains the model for future matches)
            c.execute('''
                INSERT INTO student_photos (student_id, image_path, embedding)
                VALUES (?, ?, ?)
            ''', (st['id'], db_dataset_path, unk['embedding']))

            # 3. Update unknown_faces status
            c.execute('''
                UPDATE unknown_faces
                SET status = 'resolved', resolved_student_id = ?, resolved_type = 'existing_student', resolved_by = ?
                WHERE id = ?
            ''', (st['id'], resolved_by, unknown_id))

            # 4. Mark attendance Present if class session parameters exist
            if unk['date'] and unk['period'] and unk['subject']:
                existing_att = c.execute('''
                    SELECT id FROM attendance
                    WHERE student_id = ? AND date = ? AND period = ? AND subject = ?
                ''', (st['id'], unk['date'], unk['period'], unk['subject'])).fetchone()
                if existing_att:
                    c.execute("UPDATE attendance SET status = 'Present' WHERE id = ?", (existing_att['id'],))
                else:
                    c.execute('''
                        INSERT INTO attendance (student_id, date, period, subject, status)
                        VALUES (?, ?, ?, ?, 'Present')
                    ''', (st['id'], unk['date'], unk['period'], unk['subject']))

            conn.commit()
            return jsonify({
                'status': 'success',
                'message': f"Face photo added to {st['name']} ({st['register_number']})! Model auto-trained and attendance marked Present.",
                'student': {
                    'id': st['id'],
                    'name': st['name'],
                    'register_number': st['register_number'],
                    'department': st['department'],
                    'year': st['year'],
                    'status': 'Present'
                }
            })

        elif action == 'add_new':
            name = data.get('name', '').strip()
            reg_no = data.get('register_number', '').strip()
            department = data.get('department') or unk['department']
            year = int(data.get('year') or unk['year'])

            if not name or not reg_no:
                conn.close()
                return jsonify({'status': 'error', 'message': 'Student name and register number required'}), 400

            # 1. Copy cropped face to dataset folder as primary photo
            dataset_filename = f"{reg_no}.jpg"
            dataset_dest = os.path.join(DATASET_DIR, dataset_filename)
            if os.path.exists(crop_full_path):
                shutil.copy2(crop_full_path, dataset_dest)
            db_dataset_path = f"dataset/{dataset_filename}"

            # 2. Insert into students table
            c.execute('''
                INSERT INTO students (name, register_number, department, year, image_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, reg_no, department, year, db_dataset_path))
            new_student_id = c.lastrowid

            # 3. Store embedding in student_photos table for auto-training
            c.execute('''
                INSERT INTO student_photos (student_id, image_path, embedding)
                VALUES (?, ?, ?)
            ''', (new_student_id, db_dataset_path, unk['embedding']))

            # 4. Update unknown_faces status
            c.execute('''
                UPDATE unknown_faces
                SET status = 'resolved', resolved_student_id = ?, resolved_type = 'new_student', resolved_by = ?
                WHERE id = ?
            ''', (new_student_id, resolved_by, unknown_id))

            # 5. Record attendance for newly enrolled student
            if unk['date'] and unk['period'] and unk['subject']:
                c.execute('''
                    INSERT INTO attendance (student_id, date, period, subject, status)
                    VALUES (?, ?, ?, ?, 'Present')
                ''', (new_student_id, unk['date'], unk['period'], unk['subject']))

            conn.commit()
            return jsonify({
                'status': 'success',
                'message': f"Student {name} ({reg_no}) registered with face photo! Model auto-trained and marked Present.",
                'student': {
                    'id': new_student_id,
                    'name': name,
                    'register_number': reg_no,
                    'department': department,
                    'year': year,
                    'status': 'Present'
                }
            })

        elif action == 'mark_stranger':
            notes = data.get('notes') or f"Flagged as outsider/stranger by {resolved_by}"
            c.execute('''
                UPDATE unknown_faces
                SET status = 'stranger', resolved_type = 'stranger', resolved_by = ?, notes = ?
                WHERE id = ?
            ''', (resolved_by, notes, unknown_id))
            conn.commit()
            return jsonify({
                'status': 'success',
                'message': 'Person marked as Stranger / Visitor.'
            })

        else:
            return jsonify({'status': 'error', 'message': f'Unknown action: {action}'}), 400

    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/unknown_faces', methods=['GET'])
def api_get_unknown_faces():
    """Returns list of unknown faces for live dashboards."""
    status = request.args.get('status')
    department = request.args.get('department')
    year = request.args.get('year')

    query = '''
        SELECT u.*, s.name as student_name, s.register_number as student_reg
        FROM unknown_faces u
        LEFT JOIN students s ON u.resolved_student_id = s.id
        WHERE 1=1
    '''
    params = []
    if status and status != 'all':
        query += ' AND u.status = ?'
        params.append(status)
    if department:
        query += ' AND u.department = ?'
        params.append(department)
    if year:
        query += ' AND u.year = ?'
        params.append(int(year))

    query += ' ORDER BY u.id DESC LIMIT 100'

    conn = get_db_connection()
    records = conn.execute(query, params).fetchall()
    conn.close()

    output = []
    for r in records:
        output.append({
            'id': r['id'],
            'session_id': r['session_id'],
            'department': r['department'],
            'year': r['year'],
            'subject': r['subject'],
            'period': r['period'],
            'date': r['date'],
            'crop_image_url': '/' + r['crop_image_path'].replace('\\', '/'),
            'annotated_image_url': '/' + r['annotated_image_path'].replace('\\', '/') if r['annotated_image_path'] else None,
            'original_image_url': '/' + r['original_image_path'].replace('\\', '/') if r['original_image_path'] else None,
            'status': r['status'],
            'resolved_type': r['resolved_type'],
            'resolved_by': r['resolved_by'],
            'notes': r['notes'],
            'student_name': r['student_name'],
            'student_reg': r['student_reg'],
            'created_at': r['created_at']
        })
    return jsonify({'status': 'success', 'unknown_faces': output})

@app.route('/dashboard_stats', methods=['GET'])
def dashboard_stats():
    conn = get_db_connection()
    today = datetime.date.today().isoformat()
    classes_marked = conn.execute('SELECT COUNT(DISTINCT period) FROM attendance WHERE date = ?', (today,)).fetchone()[0]
    students_present = conn.execute('SELECT COUNT(*) FROM attendance WHERE date = ? AND status = "Present"', (today,)).fetchone()[0]
    total_students = conn.execute('SELECT COUNT(*) FROM students').fetchone()[0]
    conn.close()
    return jsonify({
        'status': 'success',
        'classes_today': classes_marked or 0,
        'present_today': students_present or 0,
        'total_students': total_students or 0
    })

@app.route('/hod_dashboard', methods=['GET'])
def hod_dashboard():
    conn = get_db_connection()
    stats = conn.execute('''
        SELECT s.year,
               COUNT(DISTINCT s.id) as total_students,
               COUNT(a.id) as total_records,
               SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) as present_count
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id
        GROUP BY s.year
        ORDER BY s.year
    ''').fetchall()

    dept_records = conn.execute('''
        SELECT department, year, COUNT(*) as student_count
        FROM students
        GROUP BY department, year
    ''').fetchall()

    teachers = conn.execute("SELECT id, name, email FROM teachers WHERE role = 'teacher'").fetchall()
    teachers_data = []
    for t in teachers:
        assignments = conn.execute('SELECT * FROM teacher_assignments WHERE teacher_id = ?', (t['id'],)).fetchall()
        teachers_data.append({
            'id': t['id'],
            'name': t['name'],
            'email': t['email'],
            'assignments': [dict(a) for a in assignments]
        })

    all_students = conn.execute('SELECT * FROM students ORDER BY year, name').fetchall()

    # Query unknown faces for HOD review
    unknown_faces = conn.execute('''
        SELECT u.*, s.name as student_name, s.register_number as student_reg
        FROM unknown_faces u
        LEFT JOIN students s ON u.resolved_student_id = s.id
        ORDER BY u.id DESC
    ''').fetchall()

    unk_stats = {
        'total': len(unknown_faces),
        'pending': sum(1 for u in unknown_faces if u['status'] == 'pending'),
        'resolved': sum(1 for u in unknown_faces if u['status'] == 'resolved'),
        'stranger': sum(1 for u in unknown_faces if u['status'] == 'stranger')
    }

    # Query recent period attendance sessions for instant Excel download
    recent_sessions = conn.execute('''
        SELECT a.date, a.period, a.subject, s.department, s.year,
               COUNT(*) as total_students,
               SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) as present_count
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        GROUP BY a.date, a.period, a.subject, s.department, s.year
        ORDER BY a.date DESC, a.period DESC
        LIMIT 25
    ''').fetchall()

    conn.close()

    departments = ['CS', 'IT', 'EEE', 'MECH']
    dept_stats = {d: {1: 0, 2: 0, 3: 0, 4: 0} for d in departments}
    for r in dept_records:
        d, y, cnt = r['department'], r['year'], r['student_count']
        if d in dept_stats and y in [1, 2, 3, 4]:
            dept_stats[d][y] = cnt

    dept_totals = {d: sum(dept_stats[d].values()) for d in departments}

    return render_template('hod_dashboard.html',
                           stats=stats,
                           dept_stats=dept_stats,
                           dept_totals=dept_totals,
                           teachers=teachers_data,
                           all_students=all_students,
                           unknown_faces=unknown_faces,
                           unk_stats=unk_stats,
                           recent_sessions=recent_sessions)

# Auto-initialize database schema on startup for both Gunicorn and direct run
init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask AI Attendance Server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

