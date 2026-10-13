import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'database.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # 1. Teachers table
    c.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'teacher'
        )
    ''')

    # 2. Students table
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            register_number TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            year INTEGER NOT NULL,
            image_path TEXT NOT NULL
        )
    ''')

    # 3. Attendance table
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            period INTEGER NOT NULL,
            subject TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')

    # 4. Teacher assignments table
    c.execute('''
        CREATE TABLE IF NOT EXISTS teacher_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            department TEXT NOT NULL,
            year INTEGER NOT NULL,
            subject TEXT NOT NULL,
            period INTEGER NOT NULL,
            FOREIGN KEY (teacher_id) REFERENCES teachers (id),
            UNIQUE(teacher_id, department, year, subject, period)
        )
    ''')

    # 5. Additional Student Photos & Pre-calculated Embeddings (Auto-training)
    c.execute('''
        CREATE TABLE IF NOT EXISTS student_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')

    # 6. Unknown Face Detections (For Staff & HOD Review)
    c.execute('''
        CREATE TABLE IF NOT EXISTS unknown_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            department TEXT NOT NULL,
            year INTEGER NOT NULL,
            subject TEXT,
            period INTEGER,
            date TEXT,
            crop_image_path TEXT NOT NULL,
            original_image_path TEXT,
            annotated_image_path TEXT,
            embedding BLOB,
            status TEXT NOT NULL DEFAULT 'pending',
            resolved_student_id INTEGER,
            resolved_type TEXT,
            resolved_by TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (resolved_student_id) REFERENCES students (id)
        )
    ''')

    # Seed default accounts from paper
    c.execute("SELECT * FROM teachers WHERE email = 'hod@college.edu'")
    if not c.fetchone():
        c.execute("INSERT INTO teachers (name, email, password, role) VALUES ('HOD Admin', 'hod@college.edu', 'hod123', 'hod')")

    c.execute("SELECT * FROM teachers WHERE email = 'teacher@college.edu'")
    if not c.fetchone():
        c.execute("INSERT INTO teachers (name, email, password, role) VALUES ('Test Teacher', 'teacher@college.edu', 'password123', 'teacher')")

    conn.commit()
    conn.close()
    print("Database initialized successfully with schema matching the paper!")

if __name__ == '__main__':
    init_db()
