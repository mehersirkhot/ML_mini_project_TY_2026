
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "students.csv"
MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"

TARGET_COLUMN = "MHT-CET Score"

FEATURE_COLUMNS = [
    "Year",
    "Attendance in lecture",
    "Practical attendance",
    "Study hours",
    "Assignment completion",
    "Sleep hours",
    "Any projects done?",
    "Screen time",
    "Most used app",
    "Physical activity",
    "Socially active irl",
    "Class participation",
]

def normalize_schema(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # Remove completely empty rows and columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    return df

def load_students():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place your Google Forms CSV in the data folder "
            "and name it students.csv."
        )

    return normalize_schema(pd.read_csv(DATA_PATH))

def save_students(df):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalize_schema(df).to_csv(DATA_PATH, index=False)

def clean_dataset(df):
    df = normalize_schema(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only rows with an actual target category
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Normalize target text without changing its categories
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

    return df

def get_feature_columns(df):
    return [col for col in FEATURE_COLUMNS if col in df.columns]
