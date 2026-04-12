import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit_authenticator as stauth
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# 1. إعداد الصفحة
st.set_page_config(page_title="Radar Signal Intelligence", layout="wide")

# 2. الربط مع Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

def get_creds():
    try:
        df = conn.read(ttl=0)
        df = df.dropna(subset=['Username', 'Password'])
        creds = {"usernames": {}}
        for _, row in df.iterrows():
            creds["usernames"][str(row['Username']).strip()] = {
                "name": str(row['Name']),
                "password": str(row['Password']),
                "email": str(row.get('Email', ''))
            }
        return creds
    except:
        return {"usernames": {}}

# تحميل البيانات وتجهيز الـ Authenticator
user_creds = get_creds()
authenticator = stauth.Authenticate(
    user_creds,
    "radar_cookie",
    "radar_key_2026",
    cookie_expiry_days=30
)

# 3. نظام التسجيل والدخول
if not st.session_state.get('authentication_status'):
    tab_login, tab_reg = st.tabs(["Login", "Register New Engineer"])
    
    with tab_reg:
        # حل مشكلة الـ KeyError 'config' بإننا نستخدم Form يدوي أضمن
        with st.form("registration_form"):
            new_name = st.text_input("Full Name")
            new_user = st.text_input("Username")
            new_pw = st.text_input("Password", type="password")
            new_email = st.text_input("Email")
            submit = st.form_submit_button("Register & Sync to Cloud")
            
            if submit and new_user and new_pw:
                # تجهيز البيانات للإرسال للجوجل شيت
                new_data = pd.DataFrame([{
                    'Name': new_name,
                    'Last name': 'Engineer',
                    'Email': new_email,
                    'Username': new_user,
                    'Password': new_pw # في المشاريع الحقيقية بنعمل Hash للكلمة
                }])
                old_df = conn.read(ttl=0)
                updated_df = pd.concat([old_df, new_data], ignore_index=True)
                conn.update(data=updated_df)
                st.success("✅ Registered! Now go to Login tab.")

    with tab_login:
        authenticator.login(location='main')
        if st.session_state['authentication_status'] is False:
            st.error('Username/password is incorrect')

# 4. واجهة الرادار (بعد نجاح الدخول)
if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'sidebar')
    st.title("📡 Radar Signal Intelligence System")
    
    # دالة السبيكتروجرام
    def get_spec(sig):
        _, _, Sxx = spectrogram(sig, fs=5000)
        return 10 * np.log10(Sxx + 1e-10)

    # تحميل الموديل
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Control Panel")
        mode = st.radio("Signal Type", ["AM", "FM"])
        if st.button("Analyze Signal"):
            model = load_model()
            t = np.linspace(0, 1, 5000)
            sig = np.sin(2*np.pi*100*t) if mode == "AM" else np.sin(2*np.pi*(100*t + 10*np.sin(2*np.pi*5*t)))
            
            with col2:
                st.subheader("Spectrogram Result")
                spec = get_spec(sig)
                fig, ax = plt.subplots()
                ax.imshow(spec, aspect='auto', cmap='magma')
                st.pyplot(fig)
                
                # نتيجة الذكاء الاصطناعي
                pred = model.predict(np.resize(spec, (1, 129, 38, 1)))
                label = "AM" if np.argmax(pred) == 0 else "FM"
                st.metric("AI Prediction", label, f"{np.max(pred)*100:.2f}% Confidence")