
import streamlit as st
import pandas as pd

from utils.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from utils.train_model import (
    load_model,
    predict_student,
    train_and_select_model,
)
from utils.visualization import probability_chart


def render(df):
    st.markdown("## MHT-CET Score Prediction")
    st.caption(
        "Enter student survey responses to estimate the reported "
        "MHT-CET score category."
    )

    # Train the model if one has not been saved yet
    payload = load_model()

    with st.expander("Model Training", expanded=payload is None):
        if payload is None:
            st.info("Train the models before making predictions.")

        if st.button("Train and Compare ML Models", type="primary"):
            try:
                payload = train_and_select_model(df)
                st.session_state["trained_model"] = payload
                st.success(
                    f"Best model: {payload['model_name']}"
                )
                st.dataframe(
                    pd.DataFrame(payload["results"]),
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    payload = st.session_state.get("trained_model") or load_model()

    if payload is None:
        st.warning("Please train the model before using prediction.")
        return

    st.success(f"Active model: {payload['model_name']}")

    st.markdown("### Student Survey Information")

    # Collect only the features used by the trained model
    feature_columns = payload["feature_columns"]
    form_data = {}

    options = {
        "Year": ["1", "2", "3", "4"],
        "Attendance in lecture": [
            "Below 25%",
            "25-50%",
            "50-75%",
            "75+",
        ],
        "Practical attendance": [
            "Below 25%",
            "25-50%",
            "50-75%",
            "75+",
        ],
        "Study hours": [
            "Low (1.5 hours/day)",
            "Medium (2-4 hours/day)",
            "High (4+ hours/day)",
        ],
        "Assignment completion": [
            "Before deadline",
            "On deadline",
            "After deadline",
            "Incomplete",
        ],
        "Sleep hours": [
            "Low (5-6 hours)",
            "Medium (6-9 hours)",
            "High (9+ hours)",
        ],
        "Any projects done?": ["Yes", "No"],
        "Screen time": ["Low", "Medium", "High"],
        "Most used app": [
            "Instagram",
            "YouTube",
            "WhatsApp",
            "Other",
        ],
        "Physical activity": ["Low", "Medium", "High"],
        "Socially active irl": ["Low", "Medium", "High"],
        "Class participation": [
            "Never",
            "Sometimes",
            "Often",
            "Always",
        ],
    }

    with st.form("prediction_form"):
        left, right = st.columns(2)

        for index, column in enumerate(feature_columns):
            container = left if index % 2 == 0 else right

            with container:
                if column in options:
                    form_data[column] = st.selectbox(
                        column,
                        options[column],
                    )
                else:
                    form_data[column] = st.text_input(column)

        submitted = st.form_submit_button(
            "Predict MHT-CET Score Category",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            result = predict_student(form_data, payload)

            st.session_state["last_prediction"] = result

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]

        st.markdown("---")
        st.markdown("### Prediction Result")

        st.success(
            f"Predicted MHT-CET Score Category: "
            f"**{result['predicted_category']}**"
        )

        probabilities = result["probabilities"]

        if probabilities:
            chart = probability_chart(probabilities)

            if chart is not None:
                st.plotly_chart(
                    chart,
                    use_container_width=True,
                )

        st.caption(
            "This is a model-based estimate from survey patterns, "
            "not an official examination result."
        )
