# Personal Diary App

## Overview
The **Personal Diary App** is a secure, user-friendly application that allows users to record daily entries, track moods, and maintain a personal journal. The app leverages Python and machine learning for sentiment analysis, providing insights into user emotions over time.

---

## Key Features
- **User Authentication:** Secure login and signup functionality.
- **Diary Management:** Create, edit, view, and search diary entries.
- **Mood Tracking:** Users can log mood with each entry for emotional tracking.
- **Sentiment Analysis:** Integrated machine learning model analyzes text entries to provide mood insights.
- **Responsive Design:** HTML templates and CSS ensure a clean and intuitive user interface.

---

## Project Structure
personal_diary_app/
│
├── app.py # Main application entry point
├── migrate_add_fields.py # Database migration scripts
├── migrate_add_mood.py # Database migration scripts
├── ml_model.py # Machine learning model script
├── models/ # Trained models
│ ├── sentiment_model.joblib
│ └── tfidf_vectorizer.joblib
├── static/ # CSS, images, and static assets
│ └── style.css
├── templates/ # HTML templates for the UI
│ ├── add_entry.html
│ ├── dashboard.html
│ ├── edit_entry.html
│ ├── index.html
│ ├── login.html
│ ├── search.html
│ ├── signup.html
│ ├── single_entry.html
│ └── view_entries.html
├── .gitignore # Excluded unnecessary files like .idea, diary.db
└── README.md # Project overview and instructions

## Installation & Setup
1. **Clone the repository**
```bash
git clone https://github.com/Arymacore/Personal-Diary-App.git
cd Personal-Diary-App
Run the application
python app.py
The app will be available locally at http://127.0.0.1:5000/.

Notes
Database files and large temporary files are excluded from the repository.
The machine learning model is pre-trained and included in models/.
HTML templates are structured for easy customization and extension.

Author
Aryma Rawat (Arymacore)
