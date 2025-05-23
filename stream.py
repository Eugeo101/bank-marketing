import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# --------------------- Custom Transformer ---------------------
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in = X.shape[1]
        return self
    def transform(self, X, y=None):
        assert self.n_features_in == X.shape[1]
        return np.log(X)

# --------------------- Helper Functions ---------------------
def categorize_pdays(x):
    if x == -1: return 'Never'
    elif x <= 30: return '0-30 days'
    elif x <= 60: return '31-60 days'
    elif x <= 90: return '61-90 days'
    else: return '91+ days'

def age_band(x):
    if x <= 25: return '18-25'
    elif x <= 35: return '26-35'
    elif x <= 45: return '36-45'
    elif x <= 55: return "45-55"
    elif x <= 65: return "55-65"
    else: return "retired"

def categorize_campaign(x):
    if x == 1: return 'one time'
    elif x == 2: return "two times"
    elif x == 3: return "3 times"
    elif x == 4: return "4 times"
    elif x == 5: return "5 times"
    else: return "more than five"

def categorize_previous(x):
    if x == 0: return 'never'
    elif x == 1: return "one time"
    elif x == 2: return "two times"
    elif x == 3: return "three times"
    elif x == 4: return "four times"
    elif 5 <= x <= 10: return "5-10"
    elif 11 <= x <= 20: return "11-20"
    elif 21 <= x <= 30: return "21-30"
    elif 31 <= x <= 40: return "31-40"
    elif 41 <= x <= 50: return "41-50"
    else: return "more than 50"

# --------------------- Streamlit Multi-Page ---------------------
st.set_page_config(page_title="Bank Campaign App", layout="wide")

page = st.sidebar.radio("📄 Navigate", ["🏠 Welcome", "📊 Dataset", "🔍 Predict"])

# --------------------- Page 1: Welcome ---------------------
if page == "🏠 Welcome":
    st.title("📢 Welcome to the Bank Marketing Campaign Predictor")
    
    # Display image
    image = Image.open("pexels-expect-best-79873-351264.jpg")  # replace with your actual image file
    st.image(image, caption="Bank Marketing Campaign", use_column_width=True)

    st.markdown("## 🔍 Project Summary")
    st.write("""
    This app predicts whether a customer will subscribe to a term deposit product based on historical marketing campaign data.

    ### 🧠 Key Insights:
    - Clients who were contacted more than once in previous campaigns are more likely to subscribe.
    - Retired and students showed higher subscription rates.
    - Contact months like May and August had better conversion rates.
    - Balance and duration of last contact have a strong effect on outcome.
    """)

# --------------------- Page 2: Dataset ---------------------
elif page == "📊 Dataset":
    st.title("📂 Explore the Dataset")
    try:
        df = pd.read_csv("bank-full.csv",sep=';')  # replace with your dataset filename
        st.write("### Sample Data")
        st.dataframe(df.head())

        st.write("### Column Description")
        st.write(df.dtypes)

        st.write("### Summary Statistics")
        st.write(df.describe())

    except Exception as e:
        st.error(f"Could not load dataset: {e}")

# --------------------- Page 3: Predict ---------------------
elif page == "🔍 Predict":
    st.title("🔍 Predict Subscription")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 100)
            job = st.selectbox("Job", ['blue-collar', 'housemaid', 'technician', 'management', 'admin.',
                                       'services', 'entrepreneur', 'retired', 'self-employed',
                                       'unemployed', 'student'])
            marital = st.selectbox("Marital", ['married', 'divorced', 'single'])
            education = st.selectbox("Education", ['secondary', 'primary', 'tertiary'])

        with col2:
            default = st.selectbox("Default", ['yes', 'no'])
            housing = st.selectbox("Housing Loan", ['yes', 'no'])
            loan = st.selectbox("Personal Loan", ['yes', 'no'])
            balance = st.number_input("Balance", min_value=-3000, max_value=100500, value=0)

        with col3:
            day = st.number_input("Contact Day", min_value=0, max_value=31, value=0)
            month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
            campaign = st.number_input("Campaign Contacts", min_value=1, max_value=25, value=1)
            pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=1000, value=-1)
            previous = st.number_input("Previous Contacts", min_value=0, max_value=100, value=0)

        submit = st.form_submit_button("📈 Predict")

    if submit:
        row = pd.DataFrame([{
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'day': day,
            'month': month,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous
        }])

        # Preprocess
        row['previous'] = row['previous'].apply(categorize_previous)
        row['campaign'] = row['campaign'].apply(categorize_campaign)
        row['age'] = row['age'].apply(age_band)
        row['pdays'] = row['pdays'].apply(categorize_pdays)
        shift_amount = abs(min(row['balance'])) + 1
        row['balance'] = row['balance'].apply(lambda x: x + shift_amount)

        st.write("### Transformed Data")
        st.dataframe(row)

        # Load and Predict
        try:
            with open("lgbm_pipeline.pkl", "rb") as f:
                model = pickle.load(f)

            if hasattr(model, "predict"):
                pred = model.predict(row)
                color = "green" if pred[0] == 'yes' else "red"
                label = "✅ Subscribed" if pred[0] == 'yes' else "❌ Not Subscribed"
                st.markdown(f"<h2 style='color:{color};'>{label}</h2>", unsafe_allow_html=True)
            else:
                st.error("🚫 Invalid model object.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
