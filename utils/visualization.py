
import pandas as pd
import plotly.express as px

from utils.preprocess import TARGET_COLUMN


def score_distribution_chart(df):
    counts = df[TARGET_COLUMN].value_counts().reset_index()
    counts.columns = ["Score Category", "Number of Students"]

    return px.bar(
        counts,
        x="Score Category",
        y="Number of Students",
        color="Score Category",
        text="Number of Students",
        title="Distribution of MHT-CET Score Categories",
        template="plotly_white",
    )


def attendance_chart(df):
    return px.histogram(
        df,
        x="Attendance in lecture",
        color=TARGET_COLUMN,
        barmode="group",
        title="Lecture Attendance vs MHT-CET Score",
        template="plotly_white",
    )


def study_hours_chart(df):
    return px.histogram(
        df,
        x="Study hours",
        color=TARGET_COLUMN,
        barmode="group",
        title="Study Hours vs MHT-CET Score",
        template="plotly_white",
    )


def lifestyle_chart(df):
    columns = [
        "Sleep hours",
        "Screen time",
        "Physical activity",
    ]

    melted = df.melt(
        id_vars=[TARGET_COLUMN],
        value_vars=columns,
        var_name="Lifestyle Factor",
        value_name="Response",
    )

    return px.histogram(
        melted,
        x="Response",
        color=TARGET_COLUMN,
        facet_row="Lifestyle Factor",
        barmode="group",
        title="Lifestyle Patterns and MHT-CET Score",
        template="plotly_white",
    )


def assignment_chart(df):
    counts = (
        df.groupby(["Assignment completion", TARGET_COLUMN])
        .size()
        .reset_index(name="Number of Students")
    )

    return px.bar(
        counts,
        x="Assignment completion",
        y="Number of Students",
        color=TARGET_COLUMN,
        barmode="group",
        title="Assignment Completion vs MHT-CET Score",
        template="plotly_white",
    )


def activity_chart(df):
    columns = [
        "Any projects done?",
        "Socially active irl",
        "Class participation",
    ]

    melted = df.melt(
        id_vars=[TARGET_COLUMN],
        value_vars=columns,
        var_name="Activity",
        value_name="Response",
    )

    return px.histogram(
        melted,
        x="Response",
        color=TARGET_COLUMN,
        facet_col="Activity",
        barmode="group",
        title="Student Activities and Score Categories",
        template="plotly_white",
    )


def correlation_chart(df):
    numeric = df.select_dtypes(include="number")

    if numeric.empty or len(numeric.columns) < 2:
        return None

    correlation = numeric.corr()

    return px.imshow(
        correlation,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Between Numeric Variables",
        template="plotly_white",
    )


def probability_chart(probabilities):
    if not probabilities:
        return None

    data = pd.DataFrame({
        "Score Category": list(probabilities.keys()),
        "Probability (%)": list(probabilities.values()),
    })

    return px.bar(
        data,
        x="Score Category",
        y="Probability (%)",
        color="Score Category",
        text="Probability (%)",
        range_y=[0, 100],
        title="Prediction Probability by Score Category",
        template="plotly_white",
    )
