import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Customer Churn Prediction", layout = "wide")
st.markdown("""
<style>

.header{
background: linear-gradient(
90deg,
#2563EB,
#60A5FA
);

padding:25px;

border-radius:0px 0px 35px 35px;

text-align:center;

color:white;

font-size:42px;

font-weight:bold;

box-shadow:
0px 5px 20px rgba(0,0,0,0.20);

margin-bottom:20px;

}

.subtitle{

text-align:center;

font-size:24px;

color:#444;

margin-top:10px;

}

</style>
""",
unsafe_allow_html=True)

# Load Files

model = load("model.pkl")
encoder = load("encoder.pkl")
scaler = load("scaler.pkl")
columns = load("columns.pkl")



# Title
st.markdown(
"""
<div class="header">

📞 Telecom Customer Churn Prediction

</div>

<div class="subtitle">

<b>Fill in customer details and predict churn risk</b>

</div>
""",
unsafe_allow_html=True
)

# User Input

with st.container(border=True):

    st.subheader("👤 Customer Details")

    col1, col2,col3 = st.columns(3)

    with col1:
     SeniorCitizen = st.selectbox("Senior Citizen", [0,1], key = "Senior")
     Partner = st.selectbox("Partner",["Yes","No"], key = "Partner")
     Dependents = st.selectbox("Dependents",["Yes","No"], key = "Dependents")
     tenure = st.number_input("🕒 Tenure",min_value = 0,max_value= 72, key = "tenure")
     MultipleLines = st.selectbox("Multiple Lines",["Yes", "No", "No Phone Service"], key = "MultipleLines")
     st.caption(""" Yes: Customer uses more than one line,
                No: Customer uses only one phone line,
     No Phone Service: Customer does not subscribe to phone service""")  

    with col2:                         
     Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key = "Contract")            
     PaperlessBilling = st.selectbox("Paperless Billing",["Yes","No"], key = "Paperless")
     PaymentMethod = st.selectbox("💳 Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], key = "Payment")    
     MonthlyCharges = st.number_input("Monthly Charges", min_value = 0.0, step = 0.01, key = "Monthly")  
     TotalCharges =  st.number_input("Total Charges", min_value = 0.0, step = 0.01, key = "Total")


    with col3:  
     InternetService = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"], key = "InternetService")  

     if InternetService != "No":

      OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"], key = "Security")
      OnlineBackup = st.selectbox("Online Backup",["Yes", "No"], key = "Backup" )
      DeviceProtection = st.selectbox("Device Protection",["Yes", "No"], key = "DProtection")
      TechSupport = st.selectbox("Tech Support",["Yes","No"], key = "TechSupport")
      StreamingTV = st.selectbox("Streaming TV",["Yes","No"], key = "StreamingTV")
      StreamingMovies = st.selectbox("Streaming Movies",["Yes","No"], key = "StreamingMovies")
     else:
      OnlineSecurity = "No internet service"
      OnlineBackup = "No internet service"
      DeviceProtection = "No internet service"
      TechSupport = "No internet service"
      StreamingTV = "No internet service"
      StreamingMovies = "No internet service"


# Predict

st.markdown("""
<style>

div.stButton > button {

    background: linear-gradient(
        90deg,
        #3B82F6,
        #9333EA
    );

    color: white;

    border-radius: 12px;

    height: 55px;

    font-size: 20px;

    font-weight: bold;

    border: none;

    transition: 0.3s;

}


div.stButton > button:hover {

    transform: scale(1.05);

    box-shadow:
    0px 8px 20px
    rgba(0,0,0,0.25);

}

</style>
""", unsafe_allow_html=True)
left, center, right = st.columns([2,2,2])

with center:
    
 if st.button("Predict", use_container_width=True):
    input_df = pd.DataFrame({
        "SeniorCitizen" : [SeniorCitizen],
        "Partner" : [Partner],
        "Dependents" : [Dependents],
        "tenure" : [tenure],
        "MultipleLines" : [MultipleLines],
        "InternetService" : [InternetService],
        "OnlineSecurity" : [OnlineSecurity],
        "OnlineBackup" : [OnlineBackup],
        "DeviceProtection" : [DeviceProtection],
        "TechSupport" : [TechSupport],
        "StreamingTV" : [StreamingTV],
        "StreamingMovies" : [StreamingMovies],
        "Contract" : [Contract],
        "PaperlessBilling" : [PaperlessBilling],
        "PaymentMethod" : [PaymentMethod],
        "MonthlyCharges" : [MonthlyCharges],
        "TotalCharges" : [TotalCharges]
    })

# Encoding 

    input_df = pd.get_dummies(input_df, drop_first=True)

# Match Training

    input_df = (input_df.reindex(columns = columns, fill_value=0))

    display_df = input_df.T.reset_index()
    display_df.columns = ["Features","Value"]

# Sacle

    input_scaled = (scaler.transform(input_df))

# Predict

    pred = model.predict(input_scaled)

    result = (encoder.inverse_transform(pred))[0]

    st.subheader("📋 Processed Customer Data")
    st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True,
    height=250
)

    if result== "Yes":
      
      with center:
        st.error("⚠ Customer likely to churn")
    else:
      with center:
        st.success("✅ Customer is likely to stay")

