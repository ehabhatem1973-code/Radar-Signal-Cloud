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
conn = st.connection("gsheets", type=GSheetsConnection)

def get_all_users():
    df = conn.read(ttl=0) 
    creds = {"usernames": {}}
    for _, row in df.iterrows():
        u_name = str(row['User Name'])
        creds["usernames"][u_name] = {
            "name": str(row['Name']),
            "password": str(row['Password'])
        }
    return creds

try:
    credentials = get_all_users()
except Exception as e:
    st.error("Connection Error: Check your Google Sheet Secrets!")
    credentials = {"usernames": {}}

authenticator = stauth.Authenticate(
    credentials,
    "radar_dashboard",
    "auth_key",
    cookie_expiry_days=30
)

# --- 2. التحكم في واجهة المستخدم ---
login_placeholder = st.empty()

if not st.session_state.get('authentication_status'):
    with login_placeholder.container():
        try:
            # تم إضافة pre_authorized لإلغاء خانات غير ضرورية لو حبيت
            if authenticator.register_user(location='main'):
                # سحب البيانات اللي اتسجلت فعلياً في السيرفر
                all_users_data = st.session_state['config']['credentials']['usernames']
                
                # تحويل البيانات لجدول وتصفية الأعمدة عشان تناسب الشيت بتاعك (3 أعمدة بس)
                temp_list = []
                for uname, info in all_users_data.items():
                    temp_list.append({
                        'User Name': uname,
                        'Name': info['name'],
                        'Password': info['password']
                    })
                
                new_df = pd.DataFrame(temp_list)
                
                # ترتيب الأعمدة زي الشيت بالظبط: Name ثم User Name ثم Password
                new_df = new_df[['Name', 'User Name', 'Password']]
                
                # رفع البيانات
                conn.update(data=new_df)
                st.success('Registration successful! Please login now.')
                st.rerun() 
        except Exception as e:
            if "config" in str(e):
                st.info("System is ready. Please fill the form.")
            else:
                st.error(f"Sign-up Error: {e}")
        
        result = authenticator.login(location='main')

# فك تشفير حالة الدخول
if isinstance(st.session_state.get('authentication_status'), bool):
    authentication_status = st.session_state['authentication_status']
    name = st.session_state.get('name')
    username = st.session_state.get('username')
else:
    authentication_status = None

# --- 3. صفحة الرادار ---
if authentication_status:
    login_placeholder.empty()
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f'Welcome Engineer *{name}*')

    st.title("📡 Radar Signal Intelligence")
    st.markdown("---")

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
        
        st.subheader("1. Time Domain Representation")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(t[:500], signal[:500], color='dodgerblue')
        st.pyplot(fig1)

        with st.spinner('AI analyzing...'):
            spec = get_spec(signal)
            input_data = spec.reshape(1, 129, 38, 1)
            prediction = model.predict(input_data)
            res = ['AM', 'FM'][np.argmax(prediction)]
            
            st.subheader("2. Intelligence Analysis")
            st.metric("Predicted Class", res)

            st.subheader("3. Spectrogram")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            st.pyplot(fig2)

elif authentication_status == False:
    st.error('Username/password is incorrect')