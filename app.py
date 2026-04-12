import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit_authenticator as stauth
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# --- 1. إعداد الصفحة والربط السحابي ---
st.set_page_config(page_title="Radar Signal Cloud Intelligence", layout="wide")

# إنشاء اتصال مع Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

def get_all_users():
    try:
        # قراءة البيانات مع ضمان تحديثها من السحابة
        df = conn.read(ttl=0)
        creds = {"usernames": {}}
        if df is not None and not df.empty:
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                u_name = str(row['Username']).strip()
                if u_name and u_name != 'nan':
                    creds["usernames"][u_name] = {
                        "name": str(row['Name']),
                        "password": str(row['Password']),
                        "email": str(row.get('Email', ''))
                    }
        return creds
    except Exception:
        return {"usernames": {}}

# تحميل البيانات الحالية
credentials = get_all_users()

# نظام الحماية (Authenticator)
authenticator = stauth.Authenticate(
    credentials,
    "radar_intelligence_cookie",
    "auth_signature_key_2026",
    cookie_expiry_days=30
)

# --- 2. واجهة الدخول والتسجيل ---
if not st.session_state.get('authentication_status'):
    tab1, tab2 = st.tabs(["Login", "Register New Engineer"])
    
    with tab2:
        # خانة التسجيل: لازم تملأ كل الخانات وتضغط Register
        try:
            if authenticator.register_user(location='main'):
                # الحصول على بيانات المستخدم الجديد من الجلسة
                all_users = st.session_state['config']['credentials']['usernames']
                new_username = list(all_users.keys())[-1]
                user_info = all_users[new_username]

                # تجهيز البيانات للإرسال للجوجل شيت
                new_entry = pd.DataFrame([{
                    'Name': user_info.get('name', 'N/A'),
                    'Last name': 'Engineer',
                    'Email': user_info.get('email', 'N/A'),
                    'Username': new_username,
                    'Password': user_info.get('password', ''),
                    'Password confirmation': user_info.get('password', ''),
                    'Password hint': 'Radar Project',
                    'Captcha': 'Verified'
                }])

                # دمج البيانات الجديدة ورفعها للسحابة فوراً
                existing_df = conn.read(ttl=0)
                updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
                conn.update(data=updated_df)
                st.success('✅ Engineer Registered & Cloud Synced! Please go to Login tab.')
                st.balloons()
        except Exception as e:
            st.error(f"Error during registration: {e}")

    with tab1:
        # تسجيل الدخول
        authenticator.login(location='main')
        if st.session_state.get('authentication_status') is False:
            st.error('Username/password is incorrect')
        elif st.session_state.get('authentication_status') is None:
            st.info('Please enter your credentials to access the Radar System')

# --- 3. صفحة الرادار (تظهر فقط بعد نجاح الدخول) ---
if st.session_state.get('authentication_status'):
    # إضافة الخروج في الشريط الجانبي
    authenticator.logout('Logout', 'sidebar')
    
    with st.sidebar:
        st.success(f"Welcome, Eng. {st.session_state.get('name', 'User')}")
        st.markdown("---")
        st.subheader("🌐 System Infrastructure")
        st.info("**Environment:** Docker Container (Virtualization)")
        st.info("**Access:** FusionAccess Compatible")
        st.info("**Database:** Google Cloud Real-time")

    st.title("📡 Radar Signal Intelligence System")
    st.markdown("---")

    # وظائف معالجة الإشارات
    def get_spec(signal):
        _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Signal Generation")
        signal_option = st.selectbox("Select Modulation:", ["AM Signal", "FM Signal"])
        gen_btn = st.button("Generate & Classify 🚀")

    if gen_btn:
        model = load_my_model()
        t = np.linspace(0, 1, 5000, endpoint=False)
        
        if signal_option == "AM Signal":
            signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
        else:
            signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / 5000))
        
        with col_res:
            # رسم الموجة
            st.subheader("1. Time Domain")
            fig1, ax1 = plt.subplots(figsize=(10, 3))
            ax1.plot(t[:500], signal[:500], color='dodgerblue')
            st.pyplot(fig1)

            # تحليل الذكاء الاصطناعي ورسم السبيكتروجرام
            with st.spinner('Analyzing...'):
                spec = get_spec(signal)
                prediction = model.predict(spec.reshape(1, 129, 38, 1))
                res_label = ['AM', 'FM'][np.argmax(prediction)]
                
                st.subheader("2. Spectrogram")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                st.pyplot(fig2)

                st.subheader("3. Results")
                st.metric("Detected Modulation", res_label)