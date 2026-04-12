import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit_authenticator as stauth
from streamlit_gsheets import GSheetsConnection

# 1. إعداد الصفحة والربط
st.set_page_config(page_title="Radar Cloud Intelligence", layout="wide")
conn = st.connection("gsheets", type=GSheetsConnection)

# دالة قراءة المستخدمين (Read Only - مسموح بها للسحابة العامة)
def get_creds():
    try:
        # بنقرأ الشيت اللي في الصورة
        df = conn.read(ttl=0)
        df = df.dropna(subset=['Username', 'Password'])
        creds = {"usernames": {}}
        for _, row in df.iterrows():
            creds["usernames"][str(row['Username']).strip()] = {
                "name": str(row['Name']),
                "password": str(row['Password'])
            }
        return creds
    except:
        return {"usernames": {"admin": {"name": "Admin", "password": "123"}}}

# 2. نظام الدخول
user_creds = get_creds()
authenticator = stauth.Authenticate(user_creds, "radar_c", "key_2026")

# واجهة الدخول فقط (شيلنا التسجيل مؤقتاً عشان الـ Error)
if not st.session_state.get('authentication_status'):
    st.title("📡 Login to Radar System")
    authenticator.login(location='main')
    if st.session_state['authentication_status'] is False:
        st.error('Username/password is incorrect')
    st.info("Note: Users are managed via Google Sheets")

# 3. صفحة الرادار الكاملة
if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'sidebar')
    st.title("📡 Radar Signal Cloud Intelligence")
    
    @st.cache_resource
    def load_model():
        # تحميل ملف الـ h5 اللي رفعناه
        return tf.keras.models.load_model('signal_cnn_model.h5')

    # توليد الإشارة وعرض السبيكتروجرام
    mode = st.sidebar.radio("Signal Type", ["AM", "FM"])
    if st.button("Start AI Analysis"):
        model = load_model()
        t = np.linspace(0, 1, 5000)
        sig = (1 + 0.5*np.sin(2*np.pi*5*t)) * np.sin(2*np.pi*100*t) if mode == "AM" else np.sin(2*np.pi*(100*t + 10*np.sin(2*np.pi*5*t)))
        
        # حساب الـ Spectrogram
        _, _, Sxx = spectrogram(sig, fs=5000)
        spec = 10 * np.log10(Sxx + 1e-10)
        
        # العرض
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Spectrogram (Signal Fingerprint)")
            fig, ax = plt.subplots()
            ax.imshow(spec, aspect='auto', cmap='viridis')
            st.pyplot(fig)
        
        with col2:
            st.subheader("AI Prediction")
            # تجهيز البيانات للموديل
            input_data = np.resize(spec, (1, 129, 38, 1))
            pred = model.predict(input_data)
            label = "AM" if np.argmax(pred) == 0 else "FM"
            st.success(f"Detected: {label}")
            st.metric("Confidence", f"{np.max(pred)*100:.2f}%")