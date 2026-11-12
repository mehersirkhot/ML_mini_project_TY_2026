
from io import BytesIO

import pandas as pd
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

    st.markdown("## Visual Analytics")
    st.caption("Explore patterns in the student survey dataset.")

    if df.empty:
        st.info("No survey data available for analysis.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Score Analysis",
        "Academic Habits",
        "Lifestyle",
        "Correlation & Data",
    ])

    with tab1:
        st.markdown("### MHT-CET Score Distribution")
        st.plotly_chart(
            score_distribution_chart(df),
            use_container_width=True,
        )

        st.markdown("### Score Category Summary")

        summary = (
            df[TARGET_COLUMN]
            .value_counts()
            .rename_axis("Score Category")
            .reset_index(name="Number of Students")
        )

        summary["Percentage"] = (
            summary["Number of Students"] / len(df) * 100
        ).round(2)

        st.dataframe(summary, use_container_width=True)

    with tab2:
        st.markdown("### Lecture Attendance")
        st.plotly_chart(
            attendance_chart(df),
            use_container_width=True,
        )

        st.markdown("### Study Hours")
        st.plotly_chart(
            study_hours_chart(df),
            use_container_width=True,
        )

        st.markdown("### Assignment Completion")
        st.plotly_chart(
            assignment_chart(df),
            use_container_width=True,
        )

    with tab3:
        st.markdown("### Lifestyle Patterns")
        st.plotly_chart(
            lifestyle_chart(df),
            use_container_width=True,
        )

        st.markdown("### Student Activities")
        st.plotly_chart(
            activity_chart(df),
            use_container_width=True,
        )

    with tab4:
        st.markdown("### Numerical Correlation")

        chart = correlation_chart(df)

        if chart is not None:
            st.plotly_chart(
                chart,
                use_container_width=True,
            )
        else:
            st.info("Not enough numerical data for correlation.")

        st.markdown("### Missing Values")

        missing = (
            df.isna()
            .sum()
            .reset_index()
        )

        missing.columns = ["Column", "Missing Values"]

        missing["Missing Percentage"] = (
            missing["Missing Values"] / len(df) * 100
        ).round(2)

        st.dataframe(missing, use_container_width=True)

    st.markdown("---")
    st.markdown("### Export Survey Analytics")

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Survey Data",
        )

        summary = (
            df[TARGET_COLUMN]
            .value_counts()
            .rename_axis("Score Category")
            .reset_index(name="Number of Students")
        )

        summary.to_excel(
            writer,
            index=False,
            sheet_name="Score Summary",
        )

        missing.to_excel(
            writer,
            index=False,
            sheet_name="Missing Values",
        )

    st.download_button(
        label="Download Analytics Excel Report",
        data=output.getvalue(),
        file_name="student_academic_analytics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
