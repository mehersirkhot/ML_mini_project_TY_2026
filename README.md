# EduVista – Student Academic Intelligence Platform

EduVista is a machine learning-based web application developed using Python and Streamlit to analyse student academic habits, explore performance patterns, and predict MHT-CET score categories using survey data.

The application combines machine learning, interactive data visualisation, academic insights, and prediction capabilities in one platform.

## Features

- **Interactive Dashboard:** Displays an overview of student survey responses, academic patterns, and score category distributions.
- **Visual Analytics:** Explores relationships between attendance, study hours, assignment completion, sleep, screen time, physical activity, and reported MHT-CET score categories.
- **Machine Learning Model Training:** Implements and compares four classification algorithms.
- **Model Evaluation:** Compares algorithms using accuracy, precision, recall, F1-score, confusion matrices, and classification reports.
- **Student Prediction:** Accepts academic and lifestyle information to predict an MHT-CET score category.
- **AI Assistant:** Provides responses to questions about academic patterns and the collected dataset.
- **Dataset Management:** Provides access to survey data and dataset preview and download functionality.
- **Visual Reports:** Uses interactive charts to make academic trends easier to understand.

## Machine Learning Algorithms

EduVista implements and evaluates the following classification algorithms:

1. **Naive Bayes**
2. **Decision Tree**
3. **Random Forest**
4. **K-Nearest Neighbors (KNN)**

The algorithms are trained and evaluated using the collected survey dataset. Their performance is compared to identify the best-performing model.

The selected model is saved using Python's Pickle module for subsequent predictions.

## Dataset

The dataset was collected through Google Forms and contains 186 student responses.

### Input Features

- Year of study
- Lecture attendance
- Practical attendance
- Study hours
- Assignment completion
- Sleep hours
- Project participation
- Screen time
- Most used application
- Physical activity
- Social interaction
- Class participation

### Target Variable

**MHT-CET Score**

The target variable is divided into three categories:

- 20–40 percentile
- 40–70 percentile
- 70–95+ percentile

The application uses these categories as the target labels for its classification models.

*Note: The target categories are based on self-reported survey responses and should not be interpreted as verified examination results.*

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning algorithms |
| Plotly | Interactive visualisations |
| Matplotlib / Seaborn | Data visualisation |
| Pickle | Model persistence |

## Project Structure

```text
EduVista/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── students.csv
│
├── models/
│   └── student_model.pkl
│
├── pages/
│   ├── __init__.py
│   ├── dashboard.py
│   ├── analytics.py
│   ├── prediction.py
│   ├── reports.py
│   └── student_management.py
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── recommendation.py
│   └── visualization.py
│
└── .streamlit/
    └── config.toml
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mehersirkhot/ML_mini_project_TY_2026.git
```

### 2. Navigate to the Project Directory

```bash
cd ML_mini_project_TY_2026
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run EduVista

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Model Evaluation

The machine learning models are evaluated using:

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures how many predicted instances of a category were correct.
- **Recall:** Measures how many actual instances of a category were correctly identified.
- **F1-score:** Provides a balance between precision and recall.
- **Confusion Matrix:** Displays correct and incorrect predictions across score categories.

These evaluation metrics help compare the performance of the implemented algorithms.

## Project Objective

The primary objective of EduVista is to explore how machine learning and data analytics can be applied to student academic survey data.

The platform aims to:

- Identify patterns in student academic habits.
- Analyse factors associated with reported performance categories.
- Compare machine learning classification algorithms.
- Provide an interactive environment for academic data exploration.
- Demonstrate the practical application of machine learning in education.

## Limitations

- The dataset consists of a limited number of survey responses.
- MHT-CET score categories are self-reported.
- Model predictions are experimental and are not guaranteed examination outcomes.
- The model's performance may vary when applied to different student populations.

## Developed By

**Meher, Vaidehi, Prachi & Priyanshu**

EduVista – Student Academic Intelligence Platform

*An academic machine learning mini-project.*
