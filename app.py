import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit_authenticator as stauth
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

# --- 1. إعداد الاتصال بـ Google Sheets ---
# تأكد من وجود [connections.gsheets] في Secrets ورابط الشيت
conn = st.connection("gsheets", type=GSheetsConnection)

def get_all_users():
    # قراءة البيانات (تأكد أن الشيت مضبوط على Editor)
    df = conn.read(ttl=0) 
    creds = {"usernames": {}}
    for _, row in df.iterrows():
        u_name = str(row['Username'])
        creds["usernames"][u_name] = {
            "name": str(row['Name']),
            "password": str(row['Password']),
            "email": str(row.get('Email', ''))
        }
    return creds

# تحميل بيانات المستخدمين
try:
    credentials = get_all_users()
except Exception as e:
    st.error("Connection Error: Please check your Google Sheet Secrets!")
    credentials = {"usernames": {}}

# إعداد نظام الحماية
authenticator = stauth.Authenticate(
    credentials,
    "radar_dashboard",
    "auth_key",
    cookie_expiry_days=30
)

# --- 2. واجهة الدخول والتسجيل ---
login_placeholder = st.empty()

if not st.session_state.get('authentication_status'):
    with login_placeholder.container():
        # --- نموذج التسجيل (Sign Up) ---
        try:
            if authenticator.register_user(location='main'):
                # سحب القائمة المحدثة من الـ session_state
                all_users_dict = st.session_state['config']['credentials']['usernames']
                
                # تجهيز البيانات لتطابق أعمدة الشيت في الصورة
                temp_list = []
                for uname, info in all_users_dict.items():
                    temp_list.append({
                        'Name': info.get('name', ''),
                        'Last name': 'User', # قيمة افتراضية
                        'Email': info.get('email', 'N/A'),
                        'Username': uname,
                        'Password': info.get('password', ''),
                        'Password confirmation': info.get('password', ''),
                        'Password hint': 'Work',
                        'Captcha': 'Verified'
                    })
                
                # إنشاء DataFrame ورفعه للشيت
                new_df = pd.DataFrame(temp_list)
                conn.update(data=new_df)
                
                # --- تفعيل الدخول التلقائي والتحويل لصفحة الرادار ---
                new_user_id = list(all_users_dict.keys())[-1]
                st.session_state['authentication_status'] = True
                st.session_state['username'] = new_user_id
                st.session_state['name'] = all_users_dict[new_user_id]['name']
                
                st.success('Registration Success! Accessing Radar Intelligence...')
                st.rerun() 
        except Exception as e:
            if "config" in str(e):
                st.info("System Ready. Please fill out the registration form below.")
            else:
                st.error(f"Cloud Sync Error: {e}")
        
        # نموذج الدخول
        authenticator.login(location='main')

# التأكد من حالة الدخول
authentication_status = st.session_state.get('authentication_status')
name = st.session_state.get('name')

# --- 3. صفحة الرادار الرئيسية (تظهر بعد الدخول/التسجيل) ---
if authentication_status:
    login_placeholder.empty() # مسح واجهة الدخول
    
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f'Welcome Engineer *{name}*')

    st.title("📡 Radar Signal Intelligence")
    st.markdown("---")
    st.info("Connected to Cloud Database: **Active**")

    # معالجة الإشارة
    def get_spec(signal):
        _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    model = load_my_model()

    signal_option = st.selectbox("Select Signal Type:", ["AM Signal", "FM Signal"])

    if st.button("Generate & Classify 🚀"):
        fs = 5000
        t = np.linspace(0, 1, fs, endpoint=False)
        
        if signal_option == "AM Signal":
            signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
        else:
            signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
        
        # النتائج المرئية
        st.subheader("1. Time Domain Waveform")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(t[:500], signal[:500], color='dodgerblue')
        st.pyplot(fig1)

        with st.spinner('AI analyzing signal fingerprint...'):
            spec = get_spec(signal)
            input_data = spec.reshape(1, 129, 38, 1)
            prediction = model.predict(input_data)
            res = ['AM', 'FM'][np.argmax(prediction)]
            
            st.subheader("2. Intelligence Analysis")
            st.metric("Detected Modulation", res)

            st.subheader("3. Spectrogram Analysis")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            st.pyplot(fig2)

elif authentication_status == False:
    st.error('Username/password is incorrect')