import streamlit as st
import requests

st.set_page_config(page_title="Income Prediction Dashboard", layout="centered")

st.title("Adult Income Prediction")
st.markdown("Enter an individual's demographic and financial background below to predict their income bracket")

st.write("---")

# Buid the form layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=17, max_value=90, value=35)
    education_num = st.slider("Years of Education completed (education.num)", min_value=1, max_value=16, value=10)
    hours_per_week = st.slider("Hours Worked per Week", min_value=1, max_value=99, value=40)
    
    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ])
    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])

with col2:
    capital_gain = st.number_input("Capital Gains ($)", min_value=0, max_value=99999, value=0)
    capital_loss = st.number_input("Capital Losses ($)", min_value=0, max_value=99999, value=0)
    
    occupation = st.selectbox("Occupation", [
        "Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales", "Other-service", 
        "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", "Farming-fishing", "Tech-support", "Protective-serv"
    ])
    relationship = st.selectbox("Relationship", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"])
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "Other"])

st.write("---")

# Trigger Communication Loop on Button Click
if st.button("Predict Income Bracket", type="primary"):
    # Package UI states cleanly into JSON matching our API's expected Pydantic shape
    payload = {
        "age": age,
        "workclass": workclass,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country
    }
    
    # Point to our local FastAPI microservice engine
    API_URL = "http://127.0.0.1:8000/predict"
    
    try:
        with st.spinner("Communicating with model endpoint backend..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            income_bracket = result["income_bracket"]
            probability = result["probability_of_high_income"]
            
            # 3. Present Results Visually
            st.success("Prediction Completed Successfully!")
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                if income_bracket == ">50K":
                    st.metric(label="Predicted Income Bracket", value="> $50,000 / yr")
                else:
                    st.metric(label="Predicted Income Bracket", value="≤ $50,000 / yr")
                    
            with metric_col2:
                st.metric(label="High Income Confidence", value=f"{probability * 100:.2f}%")
                
            # Progress bar for probability representation
            st.progress(probability)
            
        else:
            st.error(f"API Backend Server Error (Status Code: {response.status_code})")
            st.json(response.json())
            
    except requests.exceptions.ConnectionError:
        st.error("Connection Refused: Could not connect to the FastAPI serving backend. Is app.py currently running on port 8000?")