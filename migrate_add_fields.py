import sqlite3

conn = sqlite3.connect('diary.db')
c = conn.cursor()

try:
    c.execute("ALTER TABLE entry ADD COLUMN title TEXT DEFAULT ''")
    print("Added 'title' column.")
except sqlite3.OperationalError:
    print("'title' column already exists, skipping.")

try:
    c.execute("ALTER TABLE entry ADD COLUMN tags TEXT")
    print("Added 'tags' column.")
except sqlite3.OperationalError:
    print("'tags' column already exists, skipping.")

conn.commit()
conn.close()
print("Migration complete!")
