import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import re
from datetime import datetime

# Session step state
if "step" not in st.session_state:
    st.session_state.step = 1

# Configure Gemini API key
try:
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
except Exception:
    st.error("Missing Gemini API key in secrets. Please configure it correctly.")

# Load and clean data
df = pd.read_csv("life_expectancy.csv")
df_cleaned = df.dropna(subset=["Country", "Year", "Life expectancy ", "Alcohol", "Schooling"])
latest_df = df_cleaned.sort_values("Year", ascending=False).groupby("Country").first().reset_index()
latest_df = latest_df[["Country", "Life expectancy ", "Alcohol", "Schooling"]]
latest_df.columns = ["Country", "LifeExpectancy", "Alcohol", "Schooling"]

DISEASE_MORTALITY_DATA = {
    "heart_disease": {
        "annual_deaths_per_100k": 162.1,
        "risk_factors": {"smoking": 2.5, "high_bp": 2.2, "high_cholesterol": 1.8, "diabetes": 2.0, "obesity": 1.6, "sedentary": 1.4}
    },
    "cancer": {
        "annual_deaths_per_100k": 146.6,
        "risk_factors": {"smoking": 3.0, "alcohol_heavy": 1.5, "family_history": 2.0, "obesity": 1.3, "age_over_50": 5.0}
    },
    "stroke": {
        "annual_deaths_per_100k": 39.0,
        "risk_factors": {"smoking": 2.0, "high_bp": 3.0, "diabetes": 1.8, "atrial_fib": 2.5, "age_over_65": 3.0}
    },
    "diabetes": {
        "annual_deaths_per_100k": 22.4,
        "risk_factors": {"obesity": 3.0, "sedentary": 2.0, "family_history": 2.5, "age_over_45": 2.0, "high_bp": 1.5}
    },
    "copd": {
        "annual_deaths_per_100k": 33.4,
        "risk_factors": {"smoking": 15.0, "secondhand_smoke": 2.0, "air_pollution": 1.5, "occupational_exposure": 1.8}
    }
}

st.set_page_config(page_title="LifeRisk.AI", page_icon="🏥", layout="wide")
st.title("🏥 LifeRisk.AI – Advanced Health Risk Assessment")
st.markdown("Comprehensive lifestyle and health analysis with disease-specific mortality risk calculations.")
st.divider()

# Step 1: Basic Info
if st.session_state.step == 1:
    st.header("📋 Step 1: Basic Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("📅 Age", min_value=18, max_value=100, value=35, key="age")
        gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"], key="gender")
        height = st.number_input("📏 Height (cm)", min_value=120, max_value=220, value=170, key="height")
        weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=70, key="weight")
    with col2:
        city = st.text_input("🌆 City", placeholder="e.g., Mumbai, Delhi", key="city")
        occupation = st.selectbox("💼 Occupation Type", ["Desk Job", "Physical Labor", "Healthcare", "Teaching", "Other"], key="occupation")
        income_bracket = st.selectbox("💰 Income Level", ["0 income", "< 5 LPA", "5-10 LPA", "10-20 LPA", "20+ LPA"], key="income_bracket")

    bmi = st.session_state.weight / ((st.session_state.height / 100) ** 2)
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25:
        bmi_status = "Normal"
    elif bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"
    st.metric("BMI", f"{bmi:.1f}", bmi_status)

    if st.button("Next ➡️", key="next1"):
        st.session_state.step = 2
        st.rerun()

# Step 2: Health History
elif st.session_state.step == 2:
    st.header("🏥 Step 2: Health History")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚬 Smoking & Substance Use")
        smoking_status = st.radio("Smoking Status", ["Never", "Former (quit)", "Current smoker"], key="smoking_status")
        if smoking_status == "Current smoker":
            cigarettes_per_day = st.slider("Cigarettes per day", 0, 40, 10)
            smoking_years = st.slider("Years smoking", 0, 50, 10)
        elif smoking_status == "Former (quit)":
            years_quit = st.slider("Years since quitting", 0, 30, 5)
            smoking_years = st.slider("Total years smoked", 0, 50, 10)
        
        alcohol_consumption = st.selectbox("🍺 Alcohol Consumption", ["Never", "Occasional", "Moderate", "Heavy"], key="alcohol_consumption")
        
        st.subheader("💊 Current Health Conditions")
        health_conditions = st.multiselect("Select any conditions you have:", ["High Blood Pressure", "High Cholesterol", "Diabetes Type 1", "Diabetes Type 2", "Heart Disease", "Asthma", "COPD", "Depression", "Anxiety", "Arthritis", "Kidney Disease"], key="health_conditions")
        
        medications = st.text_area("💉 Current Medications", placeholder="List your regular medications...", key="medications")
    
    with col2:
        st.subheader("👨‍👩‍👧‍👦 Family History")
        family_history = st.multiselect("Family history of:", ["Heart Disease", "Cancer", "Diabetes", "Stroke", "High BP", "Mental Health Issues", "Alzheimer's", "Kidney Disease"], key="family_history")
        
        st.subheader("🏃‍♂️ Lifestyle Factors")
        exercise_frequency = st.selectbox("Exercise Frequency", ["Never", "1-2 times/week", "3-4 times/week", "Daily"], key="exercise_frequency")
        exercise_intensity = st.selectbox("Exercise Intensity", ["Light", "Moderate", "Vigorous"], key="exercise_intensity")
        
        sleep_hours = st.slider("😴 Average Sleep Hours", 3, 12, 7, key="sleep_hours")
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"], key="sleep_quality")
        
        stress_level = st.slider("📈 Stress Level (1-10)", 1, 10, 5, key="stress_level")
        
        st.subheader("🍽️ Diet & Nutrition")
        diet_type = st.selectbox("Diet Type", ["Omnivore", "Vegetarian", "Vegan", "Keto", "Mediterranean"], key="diet_type")
        processed_food = st.selectbox("Processed Food Consumption", ["Rarely", "Sometimes", "Often", "Very Often"], key="processed_food")

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("⬅️ Back", key="back2"):
            st.session_state.step = 1
            st.rerun()

    with col_next:
        if st.button("Next ➡️", key="next2"):
            st.session_state.step = 3
            st.rerun()


# Step 3: Risk Analysis
elif st.session_state.step == 3:
    st.header("🔬 Step 3: Risk Analysis")

    # Gather all user inputs from session_state
    user_profile = {
    "age": st.session_state.get("age", 35),
    "gender": st.session_state.get("gender", "Male"),
    "bmi": st.session_state.get("weight", 70) / ((st.session_state.get("height", 170) / 100) ** 2),
    "smoking_status": st.session_state.get("smoking_status", "Never"),
    "health_conditions": st.session_state.get("health_conditions", []),
    "family_history": st.session_state.get("family_history", []),
    "exercise_frequency": st.session_state.get("exercise_frequency", "Never"),
    "sleep_hours": st.session_state.get("sleep_hours", 7),
    "stress_level": st.session_state.get("stress_level", 5),
    "alcohol_consumption": st.session_state.get("alcohol_consumption", "Never")
    }

    def calculate_disease_risk(disease, user_profile):
        base_risk = DISEASE_MORTALITY_DATA[disease]["annual_deaths_per_100k"] / 100000
        risk_multiplier = 1.0
        risk_factors = DISEASE_MORTALITY_DATA[disease]["risk_factors"]

        if "smoking" in risk_factors:
            if user_profile.get("smoking_status") == "Current smoker":
                risk_multiplier *= risk_factors["smoking"]
            elif user_profile.get("smoking_status") == "Former (quit)":
                risk_multiplier *= (risk_factors["smoking"] * 0.5)

        if "high_bp" in risk_factors and "High Blood Pressure" in user_profile.get("health_conditions", []):
            risk_multiplier *= risk_factors["high_bp"]

        if "high_cholesterol" in risk_factors and "High Cholesterol" in user_profile.get("health_conditions", []):
            risk_multiplier *= risk_factors["high_cholesterol"]

        if "diabetes" in risk_factors and any("Diabetes" in cond for cond in user_profile.get("health_conditions", [])):
            risk_multiplier *= risk_factors["diabetes"]

        if "obesity" in risk_factors and user_profile.get("bmi", 25) >= 30:
            risk_multiplier *= risk_factors["obesity"]

        if "sedentary" in risk_factors and user_profile.get("exercise_frequency") == "Never":
            risk_multiplier *= risk_factors["sedentary"]

        if "family_history" in risk_factors:
            disease_map = {
                "heart_disease": "Heart Disease",
                "cancer": "Cancer",
                "stroke": "Stroke",
                "diabetes": "Diabetes"
            }
            if disease_map.get(disease) in user_profile.get("family_history", []):
                risk_multiplier *= risk_factors["family_history"]

        if "age_over_50" in risk_factors and user_profile.get("age", 0) > 50:
            risk_multiplier *= risk_factors["age_over_50"]

        if "age_over_65" in risk_factors and user_profile.get("age", 0) > 65:
            risk_multiplier *= risk_factors["age_over_65"]

        return base_risk * risk_multiplier

    disease_risks = {}
    for disease in DISEASE_MORTALITY_DATA.keys():
        disease_risks[disease] = calculate_disease_risk(disease, user_profile)

    def calculate_life_expectancy(city_input, user_profile, df):
        country = "India" if "india" in city_input.lower() else "India"
        row = df[df["Country"] == country]
        base_life = row["LifeExpectancy"].values[0] if not row.empty else 72.0

        if user_profile["gender"] == "Female":
            base_life += 3

        if user_profile["smoking_status"] == "Current smoker":
            base_life -= 10
        elif user_profile["smoking_status"] == "Former (quit)":
            base_life -= 3

        if user_profile["alcohol_consumption"] == "Heavy":
            base_life -= 5
        elif user_profile["alcohol_consumption"] == "Moderate":
            base_life -= 1

        if user_profile["exercise_frequency"] in ["3-4 times/week", "Daily"]:
            base_life += 3
        elif user_profile["exercise_frequency"] == "Never":
            base_life -= 4

        if user_profile["bmi"] >= 30:
            base_life -= 3
        elif 25 <= user_profile["bmi"] < 30:
            base_life -= 1

        if user_profile["sleep_hours"] < 6:
            base_life -= 2
        elif user_profile["sleep_hours"] > 9:
            base_life -= 1

        if user_profile["stress_level"] >= 8:
            base_life -= 2

        serious_conditions = ["Heart Disease", "Diabetes Type 1", "Diabetes Type 2", "COPD"]
        condition_penalty = sum(3 for cond in user_profile["health_conditions"] if cond in serious_conditions)
        base_life -= condition_penalty

        return max(base_life, user_profile["age"] + 1)
    predicted_life = calculate_life_expectancy(st.session_state.get("city", "India"), user_profile, latest_df)
    def generate_enhanced_analysis(user_profile, disease_risks, predicted_life):
        disease_summary = "\n".join([f"- {disease.replace('_', ' ').title()}: {risk*100:.3f}% annual risk"
                                   for disease, risk in disease_risks.items()])

        prompt = f"""
        Analyze this comprehensive health profile:

        Demographics: Age {user_profile['age']}, {user_profile['gender']}, BMI {user_profile['bmi']:.1f}
        Smoking: {user_profile['smoking_status']}
        Health Conditions: {', '.join(user_profile['health_conditions']) if user_profile['health_conditions'] else 'None'}
        Family History: {', '.join(user_profile['family_history']) if user_profile['family_history'] else 'None'}
        Exercise: {user_profile['exercise_frequency']}
        Sleep: {user_profile['sleep_hours']} hours
        Stress Level: {user_profile['stress_level']}/10

        Disease-Specific Annual Mortality Risks:
        {disease_summary}

        Predicted Life Expectancy: {predicted_life} years

        Provide:
        1. A detailed 4-5 sentence risk assessment highlighting the top 3 concerns
        2. Specific recommendations to reduce the highest risks
        3. Appropriate insurance recommendations based on the risk profile
        4. Lifestyle modifications that could add years to life expectancy

        Be specific about which diseases pose the highest risk and why.
        """

        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    with st.spinner("Performing comprehensive analysis..."):
        analysis = generate_enhanced_analysis(user_profile, disease_risks, predicted_life)

    st.success("✅ Analysis Complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Disease-Specific Risk Assessment")
        sorted_risks = sorted(disease_risks.items(), key=lambda x: x[1], reverse=True)
        for disease, risk in sorted_risks:
            disease_name = disease.replace('_', ' ').title()
            annual_risk_pct = risk * 100
            if annual_risk_pct > 0.5:
                st.error(f"🔴 **{disease_name}**: {annual_risk_pct:.3f}% annual risk")
            elif annual_risk_pct > 0.1:
                st.warning(f"🟡 **{disease_name}**: {annual_risk_pct:.3f}% annual risk")
            else:
                st.info(f"🟢 **{disease_name}**: {annual_risk_pct:.3f}% annual risk")

        st.subheader("📈 Life Expectancy Prediction")
        years_remaining = max(predicted_life - st.session_state.get("age", 35), 0)
        st.metric("Estimated Life Expectancy", f"{predicted_life:.0f} years")
        st.metric("Estimated Years Remaining", f"{years_remaining:.0f} years")

    with col2:
        st.subheader("🧠 AI Health Analysis")
        st.write(analysis)

    st.subheader("💼 Personalized Insurance Recommendations")
    high_risk_diseases = [disease for disease, risk in disease_risks.items() if risk > 0.001]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🛡️ Term Life Insurance")
        coverage_amount = max(500000, user_profile["age"] * 10000)
        st.write(f"**Recommended Coverage**: ₹{coverage_amount:,}")
        st.write("Essential for income replacement and family protection")
    with col2:
        st.markdown("### 🏥 Health Insurance")
        if high_risk_diseases:
            st.write("**High Priority** - Multiple risk factors identified")
            st.write(f"Recommended: ₹10-15 lakhs coverage")
        else:
            st.write("**Standard Priority** - Low risk profile")
            st.write(f"Recommended: ₹5-10 lakhs coverage")
    with col3:
        st.markdown("### 🎯 Critical Illness Cover")
        if any(disease in ["heart_disease", "cancer", "stroke"] for disease in high_risk_diseases):
            st.write("**Highly Recommended** - Elevated risk for critical illnesses")
            st.write("Coverage: ₹25-50 lakhs")
        else:
            st.write("**Consider** - General protection")
            st.write("Coverage: ₹10-25 lakhs")

    report = f"""
🏥 Enhanced LifeRisk.AI - Comprehensive Health Risk Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERSONAL INFORMATION
====================
Age: {st.session_state.get("age", 35)}
Gender: {st.session_state.get("gender", "Male")}
BMI: {user_profile['bmi']:.1f}
City: {st.session_state.get("city", "India")}
Occupation: {st.session_state.get("occupation", "-")}

HEALTH PROFILE
==============
Current Health Conditions: {', '.join(st.session_state.get("health_conditions", [])) or 'None reported'}
Family History: {', '.join(st.session_state.get("family_history", [])) or 'None reported'}
Smoking Status: {st.session_state.get("smoking_status", "Never")}
Alcohol Consumption: {st.session_state.get("alcohol_consumption", "Never")}
Exercise Frequency: {st.session_state.get("exercise_frequency", "Never")}
Sleep Hours: {st.session_state.get("sleep_hours", 7)}
Stress Level: {st.session_state.get("stress_level", 5)}/10

DISEASE-SPECIFIC RISK ASSESSMENT
===============================
{chr(10).join([f"{disease.replace('_', ' ').title()}: {risk*100:.4f}% annual mortality risk" for disease, risk in sorted_risks])}

LIFE EXPECTANCY
===============
Predicted Life Expectancy: {predicted_life:.0f} years
Years Remaining: {years_remaining:.0f} years

AI ANALYSIS
===========
{analysis}

INSURANCE RECOMMENDATIONS
========================
1. Term Life Insurance: ₹{coverage_amount:,} coverage
2. Health Insurance: ₹{'10-15' if high_risk_diseases else '5-10'} lakhs
3. Critical Illness: ₹{'25-50' if any(d in ['heart_disease', 'cancer', 'stroke'] for d in high_risk_diseases) else '10-25'} lakhs

This report is for informational purposes only and should not replace professional medical advice.
"""

    st.download_button(
        label="📥 Download Comprehensive Health Report",
        data=report,
        file_name=f"health_risk_report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

st.divider()
if st.button("🔄 Re-Assess", key="reassess", disabled=False):
    st.session_state.step = 1
    st.rerun()


