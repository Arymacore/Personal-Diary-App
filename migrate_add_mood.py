import sqlite3
conn = sqlite3.connect('diary.db')
c = conn.cursor()
# Add column if not exists (SQLite needs workaround)
# We'll create a new table and copy data if column absent
try:
    c.execute("ALTER TABLE entries ADD COLUMN mood TEXT")
    conn.commit()
    print("Added mood column.")
except Exception as e:
    print("Could not add column (it may already exist):", e)
conn.close()
