
import streamlit as st

from utils.preprocess import clean_dataset, TARGET_COLUMN
from utils.visualization import (
    score_distribution_chart,
    attendance_chart,
    study_hours_chart,
    lifestyle_chart,
    assignment_chart,
    activity_chart,
    correlation_chart,
)


def render(df):
    df = clean_dataset(df)

    st.markdown("## Student Academic Performance Dashboard")
    st.caption(
        "Analysis of student habits, academic activities "
        "and reported MHT-CET score categories."
    )

    if df.empty:
        st.info("No valid survey responses are available.")
        return

    total_students = len(df)
    score_categories = df[TARGET_COLUMN].nunique()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Responses", total_students)
    c2.metric("Score Categories", score_categories)
    c3.metric(
        "Missing Values",
        int(df.isna().sum().sum()),
    )

    st.markdown("---")

    st.markdown("### MHT-CET Score Distribution")
    st.plotly_chart(
        score_distribution_chart(df),
        use_container_width=True,
    )

    st.markdown("### Academic Habits")

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            attendance_chart(df),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            study_hours_chart(df),
            use_container_width=True,
        )

    st.markdown("### Lifestyle Analysis")
    st.plotly_chart(
        lifestyle_chart(df),
        use_container_width=True,
    )

    st.markdown("### Assignments and Activities")

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            assignment_chart(df),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            activity_chart(df),
            use_container_width=True,
        )

    st.markdown("### Correlation Analysis")

    chart = correlation_chart(df)

    if chart is not None:
        st.plotly_chart(
            chart,
            use_container_width=True,
        )
    else:
        st.info(
            "Not enough numerical columns are available "
            "for correlation analysis."
        )

    st.markdown("### Survey Data Preview")
    st.dataframe(
        df.head(20),
        use_container_width=True,
    )
