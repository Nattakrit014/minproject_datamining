import pandas as pd
import joblib
import streamlit as st

model = joblib.load('marine_quality_model.pkl')

st.set_page_config(page_title='Coastal seawater quality forecasting model', page_icon='🌊', layout='centered')

# Apply custom CSS to style the main container with pastel colors and make it responsive
st.markdown("""
    <style>
        /* Background color for the entire page */
        .stApp {
            background-color: #E0F7FA; /* Light pastel blue for ocean feel */
        }

        /* Main container styling */
        .block-container {
            max-width: 55%;  /* Adjust width of the main container */
            margin: 0 auto;  /* Center the container */
            padding: 50px;  /* Add padding inside the container */
            border-radius: 15px;  /* Add rounded corners */
            background-color: #ffffff; /* White background for the main container */
            box-shadow: 0px 4px 8px rgba(0, 128, 128, 0.1); /* Soft shadow effect */
        }

        /* Title styling to center the text with a pastel color */
        .custom-title {
            text-align: center;  /* Center the title */
            font-size: 2.5em;    /* Adjust the font size */
            font-weight: bold;   /* Make the title bold */
            color: #006994;      /* Pastel deep sea blue */
            margin-bottom: 20px; /* Add space below the title */
        }

        /* Styling for input widgets */
        .stTextInput, .stSelectbox {
            background-color: #F0FFFF; /* Light cyan background for inputs */
            border: 1px solid #B2DFDB; /* Add a soft border */
            border-radius: 10px;  /* Rounded input fields */
        }

        /* Button styling */
        .stButton button {
            background-color: #80DEEA; /* Pastel cyan */
            color: #006064;            /* Darker cyan for text */
            font-size: 1.2em;          /* Larger button text */
            border-radius: 12px;       /* Rounded button */
            padding: 10px 20px;        /* Add padding for a larger button */
            border: none;              /* Remove button border */
        }

        .stButton button:hover {
            background-color: #00ACC1; /* Darker cyan on hover */
        }

        /* Adjusting font for general text to match theme */
        body, p, div {
            font-family: 'Arial', sans-serif;
            color: #004D40; /* Soft sea green */
        }

        /* Footer and other small elements styling */
        footer {
            text-align: center;
            color: #004D40;
        }

        /* Media query for mobile responsiveness */
        @media only screen and (max-width: 768px) {
            /* Reduce title font size for mobile */
            .custom-title {
                font-size: 1.6em;  /* Smaller title for mobile */
            }

            /* Make the input layout single-column on mobile */
            .stApp .stColumns {
                display: flex;
                flex-direction: column;
            }

            .block-container {
                max-width: 90%; /* Adjust the container width for mobile */
                padding: 50px;  /* Reduce padding for smaller screens */
            }
        }
    </style>
    """, unsafe_allow_html=True)

# Display the title with custom styling
st.markdown('<div class="custom-title">🌊 Coastal seawater quality forecasting model 🌊</div>', unsafe_allow_html=True)

def predict_quality(features):
    try:
        # Get prediction and confidence scores
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]  # Get probabilities for each class
        
        # Get the confidence score for the predicted class
        confidence_score = max(probabilities) * 100  # Convert to percentage
        return prediction, confidence_score
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Create a 2-column layout for input fields, but single-column on mobile devices
col1, col2 = st.columns(2)

with col1:
    temp = st.number_input('Temperature (อุณหภูมิ)', min_value=0.0, max_value=100.0, step=0.1)
    pH = st.number_input('pH (ค่าพีเอช)', min_value=0.0, max_value=100.0, step=0.1)
    ec = st.number_input('Electrical Conductivity (ค่าการนำไฟฟ้า)', min_value=0.0, max_value=99999.0, step=0.1)
    sali = st.number_input('Salinity (ความเค็ม)', min_value=0.0, max_value=99999.0, step=0.1)
    do = st.number_input('Dissolved Oxygen (ออกซิเจนละลาย)', min_value=0.0, max_value=99999.0, step=0.1)
    ss = st.number_input('Suspension (สารแขวนลอย)', min_value=0.0, max_value=99999.0, step=0.1)

with col2:
    pp = st.number_input('Phosphates-Phosphorus (ฟอสเฟต-ฟอสฟอรัส)', min_value=0.0, max_value=99999.0, step=0.1)
    ta = st.number_input('Total Ammonia (แอมโมเนียทั้งหมด)', min_value=0.0, max_value=99999.0, step=0.1)
    nn = st.number_input('Nitrate-Nitrogen (ไนเตรต-ไนโตรเจน)', min_value=0.0, max_value=99999.0, step=0.1)
    tc = st.number_input('Total Coliform bacteria (แบคทีเรียโคลิฟอร์มทั้งหมด)', min_value=0.0, max_value=9999999.0, step=0.1)
    au = st.number_input('Copper (ทองแดง)', min_value=0.0, max_value=100.0, step=0.1)
    zn = st.number_input('Zinc (สังกะสี)', min_value=0.0, max_value=100.0, step=0.1)

if st.button('Predict'):
    features = [temp,pH,ec,sali,do,ss,pp,ta,nn,tc,au,zn]
    prediction, confidence = predict_quality(features)
    if prediction:
        st.success(f'The level of seawater quality is : **{prediction}** with a confidence score of **{confidence:.2f}%**')
