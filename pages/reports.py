from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils.preprocess import clean_dataset
from utils.recommendation import generate_recommendations


def _student_pdf(row: pd.Series) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Student Performance Analysis Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Name: {row.student_name}", styles["Normal"]),
        Paragraph(f"Register Number: {row.register_number}", styles["Normal"]),
        Paragraph(f"Department: {row.department}", styles["Normal"]),
        Spacer(1, 12),
    ]
    table_data = [
        ["Metric", "Value"],
        ["Attendance", f"{row.attendance_percentage:.1f}%"],
        ["Internal Marks", f"{row.internal_marks:.1f}"],
        ["Assignment Score", f"{row.assignment_score:.1f}"],
        ["Study Hours / Day", f"{row.study_hours_per_day:.1f}"],
        ["Previous GPA", f"{row.previous_gpa:.2f}"],
        ["Performance", f"{row.performance_percentage:.1f}%"],
        ["Grade", row.grade],
        ["Risk Level", row.risk_level],
    ]
    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([table, Spacer(1, 12), Paragraph("Recommendations", styles["Heading2"])])
    for item in generate_recommendations(row.to_dict()):
        story.append(Paragraph(f"- {item}", styles["Normal"]))
    doc.build(story)
    return buffer.getvalue()


def _summary_pdf(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Prediction Summary Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Total Students: {len(df)}", styles["Normal"]),
        Paragraph(f"Average Performance: {df['performance_percentage'].mean():.1f}%", styles["Normal"]),
        Paragraph(f"Average Attendance: {df['attendance_percentage'].mean():.1f}%", styles["Normal"]),
        Spacer(1, 12),
    ]
    rows = [["Risk Level", "Count"]] + df["risk_level"].value_counts().reset_index().values.tolist()
    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    doc.build(story)
    return buffer.getvalue()


def render(df):
    df = clean_dataset(df)
    st.markdown("## Reports")
    if df.empty:
        st.info("No records available for report generation.")
        return

    st.download_button("Download Prediction Summary PDF", _summary_pdf(df), "prediction_summary.pdf", "application/pdf")

    selected = st.selectbox("Select Student", df["register_number"].astype(str).tolist())
    row = df[df["register_number"].astype(str).eq(selected)].iloc[0]
    st.write(f"Generating report for **{row.student_name}**")
    st.download_button("Download Student Analysis PDF", _student_pdf(row), f"{row.register_number}_analysis_report.pdf", "application/pdf")

    st.markdown("### At-Risk Follow-Up List")
    st.dataframe(df[df["risk_level"].ne("Low Risk")][["student_name", "register_number", "department", "attendance_percentage", "performance_percentage", "risk_level"]], use_container_width=True, hide_index=True)
