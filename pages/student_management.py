from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.preprocess import clean_dataset, load_students, save_students


def _form(defaults: dict | None = None, prefix: str = "student") -> dict:
    defaults = defaults or {}
    c1, c2 = st.columns(2)
    with c1:
        student_name = st.text_input("Student Name", value=str(defaults.get("student_name", "")), key=f"{prefix}_name")
        register_number = st.text_input("Register Number", value=str(defaults.get("register_number", "")), key=f"{prefix}_reg")
        department = st.text_input("Department", value=str(defaults.get("department", "Computer Science")), key=f"{prefix}_dept")
        semester = st.number_input("Semester", 1, 12, int(defaults.get("semester", 1) or 1), key=f"{prefix}_sem")
        attendance_percentage = st.number_input("Attendance Percentage", 0.0, 100.0, float(defaults.get("attendance_percentage", 80) or 80), key=f"{prefix}_att")
        internal_marks = st.number_input("Internal Marks", 0.0, 100.0, float(defaults.get("internal_marks", 70) or 70), key=f"{prefix}_int")
    with c2:
        assignment_score = st.number_input("Assignment Score", 0.0, 100.0, float(defaults.get("assignment_score", 75) or 75), key=f"{prefix}_asg")
        study_hours_per_day = st.number_input("Study Hours Per Day", 0.0, 12.0, float(defaults.get("study_hours_per_day", 3) or 3), key=f"{prefix}_study")
        previous_gpa = st.number_input("Previous GPA", 0.0, 10.0, float(defaults.get("previous_gpa", 7) or 7), key=f"{prefix}_gpa")
        behavioral_score = st.number_input("Behavioral Score", 0.0, 100.0, float(defaults.get("behavioral_score", 75) or 75), key=f"{prefix}_beh")
        extracurricular_activity_score = st.number_input("Extracurricular Activity Score", 0.0, 100.0, float(defaults.get("extracurricular_activity_score", 60) or 60), key=f"{prefix}_extra")
    return {
        "student_name": student_name,
        "register_number": register_number,
        "department": department,
        "semester": semester,
        "attendance_percentage": attendance_percentage,
        "internal_marks": internal_marks,
        "assignment_score": assignment_score,
        "study_hours_per_day": study_hours_per_day,
        "previous_gpa": previous_gpa,
        "behavioral_score": behavioral_score,
        "extracurricular_activity_score": extracurricular_activity_score,
        "math_score": float(defaults.get("math_score", internal_marks) or internal_marks),
        "science_score": float(defaults.get("science_score", internal_marks) or internal_marks),
        "english_score": float(defaults.get("english_score", assignment_score) or assignment_score),
        "computer_score": float(defaults.get("computer_score", assignment_score) or assignment_score),
    }


def render(df):
    st.markdown("## Student Management")
    df = clean_dataset(df)

    search = st.text_input("Search by name, register number, or department")
    view = df
    if search:
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        view = df[mask]

    st.dataframe(
        view.sort_values("performance_percentage", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Student Ranking")
    ranking = df.sort_values("performance_percentage", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    st.dataframe(ranking[["rank", "student_name", "register_number", "department", "performance_percentage", "grade", "risk_level"]], use_container_width=True, hide_index=True)

    tabs = st.tabs(["Add Student", "Edit Student", "Delete Student", "Profile"])

    with tabs[0]:
        new_student = _form(prefix="add")
        if st.button("Add Student", type="primary"):
            if not new_student["student_name"] or not new_student["register_number"]:
                st.error("Student name and register number are required.")
            elif new_student["register_number"] in df["register_number"].values:
                st.error("Register number already exists.")
            else:
                updated = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
                save_students(clean_dataset(updated))
                st.success("Student added. Refreshing data on next interaction.")

    with tabs[1]:
        options = df["register_number"].dropna().astype(str).tolist()
        selected = st.selectbox("Select Register Number", options, key="edit_select") if options else None
        if selected:
            record = df[df["register_number"].astype(str).eq(selected)].iloc[0].to_dict()
            edited = _form(record, prefix="edit")
            if st.button("Update Student", type="primary"):
                updated = df[df["register_number"].astype(str).ne(selected)]
                updated = pd.concat([updated, pd.DataFrame([edited])], ignore_index=True)
                save_students(clean_dataset(updated))
                st.success("Student updated.")

    with tabs[2]:
        options = df["register_number"].dropna().astype(str).tolist()
        selected_delete = st.selectbox("Student to Delete", options, key="delete_select") if options else None
        if selected_delete and st.button("Delete Student", type="secondary"):
            save_students(df[df["register_number"].astype(str).ne(selected_delete)])
            st.success("Student deleted.")

    with tabs[3]:
        options = df["register_number"].dropna().astype(str).tolist()
        selected_profile = st.selectbox("View Profile", options, key="profile_select") if options else None
        if selected_profile:
            profile = df[df["register_number"].astype(str).eq(selected_profile)].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Performance", f"{profile.performance_percentage:.1f}%")
            c2.metric("Grade", profile.grade)
            c3.metric("Risk", profile.risk_level)
            st.json(profile.to_dict())
