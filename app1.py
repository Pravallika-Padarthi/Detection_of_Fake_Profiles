
import streamlit as st

from datetime import datetime
import streamlit as st
import pickle
import pandas as pd



# Streamlit UI setup
st.set_page_config(page_title="Fake Profile Detection)", page_icon="🤖", layout="wide")

# Load Random Forest model
@st.cache_resource
def load_model():
    with open("rf.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()


# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 6px;
    }
    .stNumberInput>div>div>input {
        border-radius: 6px;
        border: 1px solid #ced4da;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1144/1144760.png", width=120)
st.sidebar.title("👤 Profile Inspector")
st.sidebar.write("This tool uses **Machine Learning** to detect fake profiles on X(formerly Twitter) based on profile stats.")

# Main Content
st.title("Detection of Fake Profiles in X(formerly Twitter")
st.write("Enter the user details below to predict whether the X(formerly Twitter) profile is **Fake** or **Genuine**.")

# Input Fields
statuses_count = st.number_input("Statuses Count", min_value=0, step=1, value=0)
followers_count = st.number_input("Followers Count", min_value=0, step=1, value=0)
friends_count = st.number_input(" Friends Count", min_value=0, step=1, value=0)
favourites_count = st.number_input("Favourites Count", min_value=0, step=1, value=0)
account_creation_date = st.date_input("Account Creation Date", min_value=datetime(2006, 3, 21), max_value=datetime(2025, 4, 28))


# Predict Button
if st.button("🔍 Predict"):
    from datetime import datetime
    today = datetime.today().date()
    account_age_days = (today - account_creation_date).days

    input_df = pd.DataFrame({
        'statuses_count': [statuses_count],
        'followers_count': [followers_count],
        'friends_count': [friends_count],
        'favourites_count': [favourites_count],
	'account_age': [account_age_days]
    })

    prediction = model.predict(input_df)
    confidence = model.predict_proba(input_df)[0][prediction[0]]

    if prediction[0] == 1:
        st.success(f" The profile is **Genuine**.")
    else:
        st.error(f" The profile is **Fake**.")


# Footer
st.markdown("---")
