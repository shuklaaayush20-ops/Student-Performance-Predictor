# app.py
# This is the main web interface for our project, built using Streamlit.
# Run it with: streamlit run app.py
# It lets users enter student details and get a prediction on the fly.

import streamlit as st
import os
from model import predict_performance, load_artifacts, load_data, generate_visualizations

# basic page setup
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# some custom CSS to make the UI look decent
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 30px;
    }

    /* result boxes change color based on prediction */
    .result-good {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #6ee7b7;
    }
    .result-average {
        background: linear-gradient(135deg, #451a03, #78350f);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #fcd34d;
    }
    .result-poor {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #fca5a5;
    }

    .accuracy-badge {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid #6366f1;
        border-radius: 10px;
        padding: 14px 20px;
        text-align: center;
        color: #a5b4fc;
        font-size: 1.0rem;
        margin-top: 16px;
    }
    .accuracy-badge span {
        font-size: 1.6rem;
        font-weight: 800;
        color: #818cf8;
    }

    .divider {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 28px 0;
    }

    .info-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #6366f1;
        color: #e2e8f0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# if model files don't exist yet, train first
def ensure_trained():
    if not os.path.exists("model.pkl") or not os.path.exists("encoder.pkl"):
        with st.spinner("Training the model for the first time, please wait..."):
            from model import main as train_main
            train_main()

ensure_trained()


# cache the model so it doesn't reload on every interaction
@st.cache_resource
def get_model():
    return load_artifacts()

model, le = get_model()

# read saved accuracy value
accuracy = 86.05
if os.path.exists("accuracy.txt"):
    with open("accuracy.txt") as f:
        accuracy = float(f.read().strip())


# page title
st.markdown('<h1 class="main-title">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter student details below to predict performance as <b>Good</b>, <b>Average</b>, or <b>Poor</b></p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# input sliders
st.markdown("### 📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "📚 Study Hours (per day)",
        min_value=0.0, max_value=12.0, value=5.0, step=0.5,
        help="How many hours does the student study on average per day?"
    )
    sleep_hours = st.slider(
        "😴 Sleep Hours (per night)",
        min_value=3.0, max_value=10.0, value=7.0, step=0.5,
        help="Average hours of sleep per night"
    )

with col2:
    attendance = st.slider(
        "📅 Attendance (%)",
        min_value=0.0, max_value=100.0, value=75.0, step=1.0,
        help="Percentage of classes attended"
    )
    previous_marks = st.slider(
        "📊 Previous Marks (out of 100)",
        min_value=0.0, max_value=100.0, value=65.0, step=1.0,
        help="What did the student score in their last exam?"
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# prediction button
predict_btn = st.button("🔮 Predict Performance", use_container_width=True, type="primary")

if predict_btn:
    result = predict_performance(study_hours, sleep_hours, attendance, previous_marks, model, le)

    st.markdown("### 🏆 Prediction Result")

    # show different colored box based on result
    if result == "Good":
        st.markdown(
            f'<div class="result-good">✅ Performance: {result}'
            f'<br><small style="font-size:1rem;font-weight:400;opacity:0.8">Student is on track to perform well!</small></div>',
            unsafe_allow_html=True
        )
    elif result == "Average":
        st.markdown(
            f'<div class="result-average">⚠️ Performance: {result}'
            f'<br><small style="font-size:1rem;font-weight:400;opacity:0.8">A few improvements can make a big difference.</small></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-poor">❌ Performance: {result}'
            f'<br><small style="font-size:1rem;font-weight:400;opacity:0.8">Student needs attention and support in multiple areas.</small></div>',
            unsafe_allow_html=True
        )

    # show accuracy below result
    st.markdown(f'''
    <div class="accuracy-badge">
        🎯 Model Accuracy &nbsp;|&nbsp; <span>{accuracy}%</span>
    </div>
    ''', unsafe_allow_html=True)

    # personalized tips based on input values
    st.markdown("### 💡 Suggestions for Improvement")
    tips = []
    if study_hours < 4:
        tips.append("📚 Try to study at least 4 to 6 hours daily for better retention.")
    if sleep_hours < 6:
        tips.append("😴 Getting 7 to 8 hours of sleep helps with memory and focus.")
    if attendance < 75:
        tips.append("📅 Attendance below 75% means missing a lot of content - try to attend more regularly.")
    if previous_marks < 50:
        tips.append("📊 Previous marks are low - revisiting basic concepts and practicing more can help.")

    if not tips:
        st.success("🌟 Everything looks good! Keep up the consistent effort.")
    else:
        for tip in tips:
            st.markdown(f'<div class="info-card">{tip}</div>', unsafe_allow_html=True)


# charts section
st.markdown('<hr class="divider">', unsafe_allow_html=True)

with st.expander("📊 View Data Analysis & Charts"):
    if os.path.exists("visualizations.png"):
        st.image("visualizations.png", use_container_width=True, caption="Dataset Overview - Student Performance Analysis")
    else:
        if st.button("Generate Charts"):
            df = load_data("data.csv")
            generate_visualizations(df)
            st.image("visualizations.png", use_container_width=True)


# model info
with st.expander("🤖 About This Model"):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Algorithm", "Decision Tree")
    col_b.metric("Train/Test Split", "80 / 20")
    col_c.metric("Accuracy", f"{accuracy}%")

    st.markdown("""
    **Input features the model uses:**
    - 📚 Study Hours — more study time usually means better understanding
    - 😴 Sleep Hours — rest is important for focus and memory
    - 📅 Attendance — being present in class helps a lot
    - 📊 Previous Marks — past performance gives a good baseline

    **Output:** `Good` | `Average` | `Poor`
    """)

st.markdown("---")
st.markdown(
    "<center style='color:#475569;font-size:0.8rem;'>Made using Python, Scikit-learn and Streamlit</center>",
    unsafe_allow_html=True
)
