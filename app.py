
import streamlit as st
import pandas as pd

from utils.preprocess import load_students, clean_dataset
from utils.train_model import train_and_select_model, load_model
from utils.recommendation import chatbot_response

from pages import dashboard, analytics, prediction


st.set_page_config(
    page_title="EduVista | Academic Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0B1020;
        color: #E8ECF8;
    }

    [data-testid="stHeader"] {
        background: #0B1020;
    }

    [data-testid="stSidebar"] {
        background: #10182B;
        border-right: 1px solid #26334D;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #F4F2FF !important;
    }

    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }

    p, label {
        color: #B8C3D9;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #192640, #131E33);
        border: 1px solid #2B3B59;
        border-radius: 16px;
        padding: 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #AAB8D5 !important;
    }

    [data-testid="stMetricValue"] {
        color: #BBAEFF !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    div.stButton > button {
        background: linear-gradient(110deg, #7865E8, #9A78F5);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(110deg, #927FF5, #B093FF);
        color: white;
        transform: translateY(-2px);
    }

    [data-testid="stTabs"] button {
        color: #AAB8D5;
        font-weight: 600;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #BBAEFF;
    }

    [data-testid="stExpander"] {
        background: #131E33;
        border: 1px solid #293956;
        border-radius: 12px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #293956;
        border-radius: 12px;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background: #17243B;
        color: #F4F2FF;
        border-color: #344563;
    }

    hr {
        border-color: #293956;
    }

    .brand-subtitle {
        color: #8D9BB8;
        font-size: 10px;
        letter-spacing: 2px;
        font-weight: 600;
    }

    .welcome-card {
        background: linear-gradient(115deg, #202C4A, #302B57);
        border: 1px solid #454274;
        padding: 25px;
        border-radius: 18px;
        margin-bottom: 22px;
    }

    .welcome-card h2 {
        margin: 0;
        font-size: 26px;
    }

    .welcome-card p {
        margin: 8px 0 0 0;
        color: #C0C7E0;
    }

    .sidebar-footer {
        color: #74839F;
        font-size: 11px;
        padding-top: 25px;
        border-top: 1px solid #293956;
        margin-top: 25px;
    }

    </style>
    """, unsafe_allow_html=True)


def sidebar_navigation():

    with st.sidebar:

        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="
                background:linear-gradient(135deg,#7865E8,#A18BFF);
                width:44px;height:44px;border-radius:13px;
                display:flex;align-items:center;
                justify-content:center;font-size:23px;">
                🎓
            </div>
            <div>
                <div style="
                    font-family:'Space Grotesk';
                    font-size:23px;font-weight:700;
                    color:#F4F2FF;">
                    EduVista
                </div>
                <div class="brand-subtitle">
                    ACADEMIC INTELLIGENCE
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown(
            "<p style='font-size:11px;letter-spacing:1.5px;"
            "color:#74839F;font-weight:700;'>WORKSPACE</p>",
            unsafe_allow_html=True,
        )

        pages = [
            "Dashboard",
            "Visual Analytics",
            "Model Training",
            "Prediction",
            "AI Assistant",
            "Dataset",
        ]

        icons = {
            "Dashboard": "◈",
            "Visual Analytics": "▦",
            "Model Training": "✧",
            "Prediction": "◎",
            "AI Assistant": "✳",
            "Dataset": "▤",
        }

        selected = st.radio(
            "Navigation",
            pages,
            label_visibility="collapsed",
            format_func=lambda page: f"{icons[page]}   {page}",
        )

        st.markdown("""
        <div class="sidebar-footer">
    <b style="color:#BBAEFF;">EDUVISTA</b><br>
    Student Academic Intelligence<br><br>
    Made with ♥ by<br>
    Meher, Vaidehi, Prachi & Priyanshu
</div>
        """, unsafe_allow_html=True)

    return selected


def main():

    apply_custom_style()

    page = sidebar_navigation()

    try:
        df = clean_dataset(load_students())

    except Exception as exc:
        st.error(f"Dataset loading failed: {exc}")
        st.info(
            "Check that students.csv is present inside the data folder."
        )
        return

    if page == "Dashboard":
        dashboard.render(df)

    elif page == "Visual Analytics":
        analytics.render(df)

    elif page == "Model Training":

        st.markdown("""
        <div class="welcome-card">
            <h2>Machine Learning Studio</h2>
            <p>Train, compare and evaluate classification algorithms.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Train All Models", type="primary"):

            with st.spinner("Training machine learning models..."):

                try:
                    payload = train_and_select_model(df)

                    st.session_state["trained_model"] = payload

                    st.success(
                        f"Training completed. Best model: "
                        f"{payload['model_name']}"
                    )

                    st.markdown("### Model Performance Comparison")

                    st.dataframe(
                        pd.DataFrame(payload["results"]),
                        use_container_width=True,
                    )

                    st.markdown("### Confusion Matrix")

                    matrix = pd.DataFrame(
                        payload["confusion_matrix"],
                        index=payload["class_names"],
                        columns=payload["class_names"],
                    )

                    st.dataframe(
                        matrix,
                        use_container_width=True,
                    )

                    st.markdown("### Classification Report")

                    report = pd.DataFrame(
                        payload["classification_report"]
                    ).transpose()

                    st.dataframe(
                        report,
                        use_container_width=True,
                    )

                except Exception as exc:
                    st.error(f"Model training failed: {exc}")

        else:

            payload = load_model()

            if payload:
                st.info(
                    f"Previously trained model: "
                    f"{payload['model_name']}"
                )

    elif page == "Prediction":
        prediction.render(df)

    elif page == "AI Assistant":

        st.markdown("""
        <div class="welcome-card">
            <h2>Academic Insights Assistant</h2>
            <p>Explore your student survey data through natural-language questions.</p>
        </div>
        """, unsafe_allow_html=True)

        question = st.chat_input(
            "Ask about attendance, study habits, score categories..."
        )

        if question:

            answer = chatbot_response(question, df)

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                st.write(answer)

    elif page == "Dataset":

        st.title("Survey Data Explorer")

        st.caption(
            "Explore and download the student survey responses."
        )

        c1, c2 = st.columns(2)

        c1.metric("Total Responses", len(df))
        c2.metric("Dataset Features", len(df.columns))

        st.dataframe(
            df,
            use_container_width=True,
        )

        st.download_button(
            "Download Dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="eduvista_student_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
