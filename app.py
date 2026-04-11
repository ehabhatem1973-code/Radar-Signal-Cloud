import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit_authenticator as stauth
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

# --- 1. إعداد الصفحة والربط السحابي ---
st.set_page_config(page_title="Radar Signal Intelligence", layout="wide")

# إنشاء اتصال مع Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

def get_all_users():
    try:
        # قراءة البيانات بدون كاش
        df = conn.read(ttl=0)
        creds = {"usernames": {}}
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

# تحميل بيانات المستخدمين
credentials = get_all_users()

# إعداد نظام الحماية
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
        try:
            if authenticator.register_user(location='main'):
                all_users_dict = st.session_state['config']['credentials']['usernames']
                new_username = list(all_users_dict.keys())[-1]
                user_info = all_users_dict[new_username]

                # تجهيز السطر للشيت
                new_entry = pd.DataFrame([{
                    'Name': user_info.get('name', ''),
                    'Last name': 'Engineer',
                    'Email': user_info.get('email', 'N/A'),
                    'Username': new_username,
                    'Password': user_info.get('password', ''),
                    'Password confirmation': user_info.get('password', ''),
                    'Password hint': 'Radar Project',
                    'Captcha': 'Verified'
                }])

                existing_df = conn.read(ttl=0).dropna(how='all')
                updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
                conn.update(data=updated_df)
                st.success('✅ Cloud Sync Success! Now go to Login tab.')
        except Exception as e:
            st.info("Fill the form to register.")

    with tab1:
        # التعديل المعدل لفك مشكلة الـ TypeError
        authenticator.login(location='main')
        
        if st.session_state.get('authentication_status') is False:
            st.error('Username/password is incorrect')
        elif st.session_state.get('authentication_status') is None:
            st.warning('Please enter your credentials')

# --- 3. صفحة الرادار الرئيسية (بعد نجاح الدخول) ---
if st.session_state.get('authentication_status'):
    # شريط جانبي للمهندس
    authenticator.logout('Logout', 'sidebar')
    # إضافة حماية بسيطة لو لم يتم تعريف st.session_state.get('name')
    st.sidebar.success(f"Engineer: {st.session_state.get('name', 'User')}")
    st.sidebar.markdown("---")
    st.sidebar.info("System Status: **Active** 📡")

    st.title("📡 Radar Signal Intelligence & Classification")
    st.markdown("---")

    # وظائف معالجة الإشارة
    def get_spec(signal):
        _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        # Normalization
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    @st.cache_resource
    def load_my_model():
        # تأكد من وجود ملف الموديل في نفس الفولدر على GitHub
        return tf.keras.models.load_model('signal_cnn_model.h5')

    # واجهة التحكم في الإشارات
    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Signal Generation")
        signal_option = st.selectbox("Select Modulation Type:", ["AM Signal", "FM Signal"])
        gen_btn = st.button("Generate & Classify 🚀")

    if gen_btn:
        model = load_my_model()
        fs = 5000
        t = np.linspace(0, 1, fs, endpoint=False)
        
        # توليد الإشارة بناءً على النوع المختار
        if signal_option == "AM Signal":
            signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
        else:
            signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
        
        with col_res:
            # 1. رسم الموجة في الزمن
            st.subheader("1. Time Domain (Waveform)")
            fig1, ax1 = plt.subplots(figsize=(10, 3))
            ax1.plot(t[:500], signal[:500], color='dodgerblue', linewidth=1)
            ax1.set_title(f"Generated signal: {signal_option}")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

            # 2. تحليل الذكاء الاصطناعي
            with st.spinner('Running CNN Intelligence Analysis...'):
                spec = get_spec(signal)
                # تجهيز البيانات للموديل (Input Shape: 1, 129, 38, 1)
                input_data = spec.reshape(1, 129, 38, 1)
                prediction = model.predict(input_data)
                res_label = ['AM', 'FM'][np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                st.subheader("2. Intelligence Results")
                c1, c2 = st.columns(2)
                c1.metric("Detected Modulation", res_label)
                c2.metric("Confidence Score", f"{confidence:.2f}%")

            # 3. رسم السبيكتروجرام (Fingerprint Graph) 🎨
            st.subheader("3. Spectrogram (Signal Fingerprint)")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            # استخدام cmap ملون مثل viridis لإظهار شدة الإشارة
            img = ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            ax2.set_ylabel("Frequency Bin")
            ax2.set_xlabel("Time Bin")
            ax2.set_title("Signal Power Spectrogram")
            
            # إضافة Colorbar لإظهار مقياس الشدة
            fig2.colorbar(img, ax=ax2, label='Normalized Intensity')
            st.pyplot(fig2)