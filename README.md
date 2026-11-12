# Student Performance Prediction System Using Machine Learning

A complete Streamlit web application for predicting student academic performance, identifying at-risk students, managing student records, training machine learning models, and generating analytics reports.

## Features

- Admin and teacher login with Streamlit session management
- Student CRUD management using CSV storage
- Dataset upload, preview, cleaning, and normalization
- ML model training with Linear Regression, Decision Tree, and Random Forest Classifier
- Automatic model comparison and saved model loading via pickle
- Real-time prediction of pass/fail, grade, performance percentage, risk level, and probability
- AI recommendation engine and teacher chatbot assistant
- Dashboard metrics, at-risk alerts, ranking, subject analysis, and semester forecasting
- Interactive Plotly charts plus Seaborn correlation heatmap
- PDF student reports and prediction summary reports
- Excel analytics export
- Optional voice note capture panel

## Folder Structure

```text
student_performance_project/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── student_model.pkl
├── data/
│   └── students.csv
├── pages/
│   ├── dashboard.py
│   ├── prediction.py
│   ├── analytics.py
│   ├── reports.py
│   └── student_management.py
├── utils/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── recommendation.py
│   └── visualization.py
└── assets/
```

## Installation

```bash
cd student_performance_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows:

```bash
cd student_performance_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Login

- Admin: `admin` / `admin123`
- Teacher: `teacher` / `teacher123`

For real deployment, replace the demo credentials with hashed passwords stored in environment variables or a secure database.

## Model Training

Open **ML Training** or **Prediction** and click the training button. The app trains:

- Linear Regression
- Decision Tree Regressor
- Random Forest Classifier

The model with the best pass/fail accuracy is saved to:

```text
models/student_model.pkl
```

## Dataset

The sample dataset is included at:

```text
data/students.csv
```

Required input fields:

- Student Name
- Register Number
- Department
- Attendance Percentage
- Internal Marks
- Assignment Score
- Study Hours Per Day
- Previous GPA
- Behavioral Score
- Extracurricular Activity Score

The app automatically computes missing performance, pass/fail, grade, and risk fields.

## Notes

CSV storage is used by default for easy setup. The project is organized so SQLite can be added later by replacing `load_students` and `save_students` in `utils/preprocess.py`.
