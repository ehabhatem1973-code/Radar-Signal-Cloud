import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit_authenticator as stauth
from datetime import datetime 
import os

# --- 1. إعداد البيانات الأساسية ---
names = ["Ehab Hatem", "Guest User"]
usernames = ["ehab", "guest"]
passwords = ["123", "456"] 

# تجهيز البيانات للمكتبة (نسخة 0.4.2)
credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "password": passwords[i]
        } for i in range(len(usernames))
    }
}

# تعريف نظام الصلاحيات
authenticator = stauth.Authenticate(
    credentials,
    "radar_dashboard",
    "auth_key",
    cookie_expiry_days=30
)

# --- 2. نظام التسجيل (Sign Up) ---
# بيظهر قبل اللوجن عشان لو حد جديد عايز يسجل
try:
    if authenticator.register_user(location='main'):
        st.success('User registered successfully! You can now login.')
except Exception as e:
    st.error(f"Sign-up Error: {e}")

# --- 3. نظام تسجيل الدخول (Login) ---
result = authenticator.login(location='main')

# فك تشفير النتيجة لضمان عمل الكود
if isinstance(result, tuple):
    name, authentication_status, username = result
else:
    authentication_status = st.session_state.get('authentication_status')
    name = st.session_state.get('name')
    username = st.session_state.get('username')

# --- 4. تشغيل البرنامج في حالة نجاح الدخول ---
if authentication_status:
    # تسجيل الدخول في ملف Log
    with open("logins.txt", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"User: {name} ({username}) | Login Time: {now}\n")
    
    # واجهة الخروج والترحيب في الجنب
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f'Welcome Engineer *{name}*')

    # دالة الـ Spectrogram
    def get_spec(signal):
        _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    # تحميل الموديل (مرة واحدة فقط)
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    model = load_my_model()

    # واجهة مستخدم الرادار
    st.title("📡 Radar Signal Intelligence")
    st.write("Welcome, Engineer! Select a signal type and click Generate.")

    signal_option = st.selectbox("Select Signal Type:", ["AM Signal", "FM Signal"])

    if st.button("Generate & Classify 🚀"):
        fs = 5000
        t = np.linspace(0, 1, fs, endpoint=False)
        
        if signal_option == "AM Signal":
            signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
        else:
            signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
        
        # رسم الموجة الزمنية
        fig1, ax1 = plt.subplots()
        ax1.plot(t[:500], signal[:500])
        ax1.set_title(f"Generated {signal_option}")
        st.pyplot(fig1)

        with st.spinner('Analyzing...'):
            try:
                # معالجة الإشارة
                spec = get_spec(signal)
                input_data = spec.reshape(1, 129, 38, 1)
                
                # التوقع بالذكاء الاصطناعي
                prediction = model.predict(input_data)
                classes = ['AM', 'FM']
                res = classes[np.argmax(prediction)]
                conf = np.max(prediction) * 100

                st.success(f"### Prediction: {res}")
                st.info(f"### Confidence: {conf:.2f}%")
                
                # رسم الـ Spectrogram
                fig2, ax2 = plt.subplots()
                ax2.imshow(spec, aspect='auto', origin='lower')
                ax2.set_title("Signal Spectrogram (AI View)")
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Logic Error: {e}")

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password or register below')