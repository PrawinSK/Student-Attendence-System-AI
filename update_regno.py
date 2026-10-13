import os
import sqlite3
from database import get_db_connection

conn = get_db_connection()
c = conn.cursor()

old = c.execute("SELECT * FROM students WHERE name = 'Prawin'").fetchone()
if old:
    old_img = old['image_path']
    new_img = 'dataset/950523205039.jpg'
    
    base_dir = os.path.dirname(__file__)
    old_full = os.path.join(base_dir, old_img)
    new_full = os.path.join(base_dir, new_img)
    if os.path.exists(old_full):
        if os.path.exists(new_full):
            os.remove(new_full)
        os.rename(old_full, new_full)
    
    c.execute("UPDATE students SET register_number = ?, image_path = ? WHERE id = ?",
              ('950523205039', new_img, old['id']))
    conn.commit()
    print("Database and dataset file updated successfully!")
    print("Name: Prawin")
    print("Register Number: 950523205039")
    print(f"Image: {new_img}")
else:
    print("Prawin not found in database.")

conn.close()
