pip install streamlit pandas numpy xgboost shap plotly scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DATA INTEGRATION & SIMULATION
# ==========================================
@st.cache_data
def generate_student_data(n_students=1000):
    """Simulates a unified Data Lake combining SIS, LMS, and Placement data."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'Student_ID': [f'STU{str(i).zfill(4)}' for i in range(1, n_students + 1)],
        'CGPA': np.round(np.random.normal(7.2, 1.5, n_students).clip(3.0, 10.0), 2),
        'Attendance_Pct': np.round(np.random.normal(75, 15, n_students).clip(30, 100), 1),
        'LMS_Logins_Weekly': np.random.poisson(lam=5, size=n_students).clip(0, 20),
        'Assignments_Missed': np.random.poisson(lam=2, size=n_students).clip(0, 10),
        'Aptitude_Score': np.round(np.random.normal(60, 20, n_students).clip(0, 100), 1),
        'Extracurriculars': np.random.randint(0, 5, n_students),
        'Active_Backlogs': np.random.poisson(lam=0.5, size=n_students).clip(0, 5)
    })
    
    # Define Ground Truth (Risk Rule) with some injected noise for the ML model to learn
    # A student is fundamentally at risk if they have high backlogs, low attendance, OR low CGPA
    risk_condition = (
        (data['CGPA'] < 6.0) | 
        (data['Attendance_Pct'] < 65) | 
        (data['Active_Backlogs'] >= 2)
    )
    
    # Add 5% noise to make the ML task realistic
    noise = np.random.choice([True, False], size=n_students, p=[0.05, 0.95])
    data['At_Risk_Actual'] = np.where(risk_condition ^ noise, 1, 0)
    
    return data

# ==========================================
# 2. PREDICTIVE AI & SUCCESS SCORE (XGBOOST)
# ==========================================
@st.cache_resource
def train_model(df):
    """Trains an XGBoost model to predict risk and calculate the Success Score."""
    features = ['CGPA', 'Attendance_Pct', 'LMS_Logins_Weekly', 'Assignments_Missed', 
                'Aptitude_Score', 'Extracurriculars', 'Active_Backlogs']
    
    X = df[features]
    y = df['At_Risk_Actual']
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    # Success Score is 100 - (Risk Probability * 100)
    risk_probs = model.predict_proba(X)[:, 1]
    df['Success_Score'] = np.round((1 - risk_probs) * 100, 1)
    df['Predicted_Risk_Flag'] = model.predict(X)
    
    # Segmentation Logic (Bonus Feature)
    conditions = [
        (df['CGPA'] >= 7.5) & (df['Aptitude_Score'] < 50), # High Academics, Low Placement
        (df['CGPA'] < 6.5) & (df['Attendance_Pct'] >= 80), # Trying but failing (Hidden Struggler)
        (df['Success_Score'] < 40),                        # High Risk Baseline
        (df['Success_Score'] >= 80)                        # On Track
    ]
    choices = ['Academic Introvert', 'Hidden Struggler', 'High Risk (Urgent)', 'On Track']
    df['Segment'] = np.select(conditions, choices, default='Needs Monitoring')
    
    return model, df, X

# ==========================================
# 3. EXPLAINABLE AI (SHAP WATERFALL)
# ==========================================
def plot_shap_waterfall(model, X, student_idx):
    """Generates an explainable score breakdown using SHAP values and Plotly."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Extract values for the specific student
    student_shaps = shap_values[student_idx]
    base_value = explainer.expected_value
    
    # Create Plotly Waterfall
    features = X.columns.tolist()
    
    fig = go.Figure(go.Waterfall(
        name="Student", orientation="h", measure=["relative"] * len(features) + ["total"],
        y=features + ["Final Risk Logits"],
        x=list(student_shaps) + [base_value + sum(student_shaps)],
        connector={"line":{"color":"rgb(63, 63, 63)"}},
        decreasing={"marker":{"color":"#2ca02c"}}, # Green means it reduces risk
        increasing={"marker":{"color":"#d62728"}}, # Red means it increases risk
        totals={"marker":{"color":"#1f77b4"}}
    ))
    
    fig.update_layout(
        title="Explainable AI: Drivers of Student Risk",
        waterfallgap=0.3,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ==========================================
# 4. STREAMLIT DASHBOARD (UI)
# ==========================================
st.set_page_config(page_title="Smart Campus Analytics", layout="wide")

# Initialize Data & Models
df = generate_student_data()
model, df_scored, X = train_model(df)

# Header
st.title("?? Smart Campus Analytics Platform")
st.markdown("Predict, Optimize & Improve Student Success with Explainable AI.")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", len(df_scored))
col2.metric("Students At Risk", len(df_scored[df_scored['Predicted_Risk_Flag'] == 1]))
col3.metric("Avg Success Score", f"{df_scored['Success_Score'].mean():.1f}/100")
col4.metric("Active Segments", df_scored['Segment'].nunique())

st.divider()

# Main Dashboard Layout
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Student Selection")
    # Search / Filter
    segment_filter = st.selectbox("Filter by Student Segment", df_scored['Segment'].unique())
    filtered_df = df_scored[df_scored['Segment'] == segment_filter]
    
    selected_student = st.selectbox("Select Student ID", filtered_df['Student_ID'].tolist())
    
    student_data = filtered_df[filtered_df['Student_ID'] == selected_student].iloc[0]
    student_idx = df_scored[df_scored['Student_ID'] == selected_student].index[0]
    
    # Profile Card
    st.info(f"**Selected Student:** {selected_student}\n\n**Segment:** {student_data['Segment']}")
    
    # Success Score Gauge
    score = student_data['Success_Score']
    color = "green" if score >= 80 else "orange" if score >= 60 else "red"
    st.markdown(f"### Success Score: <span style='color:{color}'>{score}/100</span>", unsafe_allow_html=True)
    
    if student_data['Predicted_Risk_Flag'] == 1:
        st.error("?? HIGH RISK FLAG DETECTED - Intervention Recommended")
    else:
        st.success("? On Track")

with right_col:
    st.subheader("Explainable Score Breakdown")
    st.markdown("The chart below shows exactly *why* the AI assigned this score. Bars pushing to the right increase the risk of failure; bars pushing to the left decrease risk.")
    
    # Render SHAP Waterfall
    shap_fig = plot_shap_waterfall(model, X, student_idx)
    st.plotly_chart(shap_fig, use_container_width=True)
    
    st.subheader("Unified Data View")
    # Display raw metrics for the student
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("CGPA", student_data['CGPA'])
    metrics_cols[1].metric("Attendance", f"{student_data['Attendance_Pct']}%")
    metrics_cols[2].metric("LMS Weekly Logins", student_data['LMS_Logins_Weekly'])
    metrics_cols[3].metric("Active Backlogs", student_data['Active_Backlogs'])
    
    metrics_cols2 = st.columns(4)
    metrics_cols2[0].metric("Aptitude Score", f"{student_data['Aptitude_Score']}/100")
    metrics_cols2[1].metric("Missed Assignments", student_data['Assignments_Missed'])
    metrics_cols2[2].metric("Extracurriculars", student_data['Extracurriculars'])

st.divider()
st.subheader("Campus Overview (Data Integration)")
st.dataframe(df_scored[['Student_ID', 'Success_Score', 'Segment', 'CGPA', 'Attendance_Pct', 'Predicted_Risk_Flag']].head(100), use_container_width=True)
