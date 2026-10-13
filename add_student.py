import os
import shutil
import sqlite3
from database import DB_PATH, get_db_connection
from face_engine import encode_student_image

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

def add_student(name, register_number, department, year, source_image_path):
    if not os.path.exists(source_image_path):
        print(f"Error: Image not found at {source_image_path}")
        return False

    os.makedirs(DATASET_DIR, exist_ok=True)
    ext = os.path.splitext(source_image_path)[1] or '.jpg'
    dest_filename = f"{register_number}{ext}"
    dest_path = os.path.join(DATASET_DIR, dest_filename)

    shutil.copy2(source_image_path, dest_path)
    db_image_path = os.path.join('dataset', dest_filename).replace('\\', '/')

    # Verify face encoding can be extracted
    embedding = encode_student_image(dest_path)
    if embedding is None:
        print(f"Warning: No clear face detected in {dest_path}. Student added, but ensure photo has a clear face.")

    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO students (name, register_number, department, year, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, register_number, department, int(year), db_image_path))
        conn.commit()
        print(f"Successfully registered {name} ({register_number})!")
        return True
    except sqlite3.IntegrityError:
        print(f"Error: Student with register number {register_number} already exists.")
        return False
    finally:
        conn.close()

if __name__ == '__main__':
    print("--- Add New Student ---")
    name = input("Enter student name: ")
    reg_no = input("Enter register number: ")
    dept = input("Enter department (e.g., CS, IT): ")
    year = int(input("Enter year (e.g., 1, 2, 3, 4): "))
    img_path = input("Enter absolute path to their photo: ").strip('"\'')
    add_student(name, reg_no, dept, year, img_path)
