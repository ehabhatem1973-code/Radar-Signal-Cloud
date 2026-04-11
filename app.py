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

credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "password": passwords[i]
        } for i in range(len(usernames))
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "radar_dashboard",
    "auth_key",
    cookie_expiry_days=30
)

# --- 2. التحكم في واجهة المستخدم (The Trick) ---
# بنعمل مكان فاضي نقدر نتحكم في ظهوره واختفائه
login_placeholder = st.empty()

# بنعرض شاشة الدخول والـ Register جوه المكان الفاضي ده
with login_placeholder.container():
    try:
        if authenticator.register_user(location='main'):
            st.success('User registered successfully! You can now login.')
    except Exception as e:
        st.error(f"Sign-up Error: {e}")
    
    result = authenticator.login(location='main')

# فك تشفير نتيجة الدخول
if isinstance(result, tuple):
    name, authentication_status, username = result
else:
    authentication_status = st.session_state.get('authentication_status')
    name = st.session_state.get('name')
    username = st.session_state.get('username')

# --- 3. تشغيل البرنامج في حالة نجاح الدخول ---
if authentication_status:
    # السطر ده بيمسح كل اللي كان جوه login_placeholder (يعني بيخفي شاشة الدخول)
    login_placeholder.empty()
    
    # تسجيل الدخول في ملف Log
    with open("logins.txt", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"User: {name} ({username}) | Login Time: {now}\n")
    
    # واجهة الخروج والترحيب في الجنب (Sidebar)
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title("🛠️ Control Panel")
    st.sidebar.success(f'Welcome Engineer *{name}*')

    # --- بداية صفحة الرادار الجديدة ---
    st.title("📡 Radar Signal Intelligence")
    st.markdown("---") # خط فاصل للتنظيم
    st.write("Current Status: **Active** | System: **Cloud-Based AI**")

    # دالة الـ Spectrogram
    def get_spec(signal):
        _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    # تحميل الموديل
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    model = load_my_model()

    # اختيار نوع الإشارة
    signal_option = st.selectbox("Select Signal Type to Analyze:", ["AM Signal", "FM Signal"])

    if st.button("Generate & Classify 🚀"):
        fs = 5000
        t = np.linspace(0, 1, fs, endpoint=False)
        
        if signal_option == "AM Signal":
            signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
        else:
            signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
        
        # عرض الموجة الزمنية
        st.subheader("1. Time Domain Representation")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t[:500], signal[:500], color='blue')
        ax1.set_title(f"Generated {signal_option}")
        st.pyplot(fig1)

        with st.spinner('AI is Analyzing Signal Characteristics...'):
            try:
                spec = get_spec(signal)
                input_data = spec.reshape(1, 129, 38, 1)
                
                prediction = model.predict(input_data)
                classes = ['AM', 'FM']
                res = classes[np.argmax(prediction)]
                conf = np.max(prediction) * 100

               
                # رسم الـ Spectrogram
                st.subheader("2. Spectrogram (Signal Fingerprint)")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                ax2.set_title("Signal Spectrogram (CNN Input)")
                st.pyplot(fig2)

               # عرض النتيجة بشكل شيك
                st.subheader("3. Intelligence Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"### Prediction: {res}")
                with col2:
                    st.info(f"### Confidence: {conf:.2f}%")
                

            except Exception as e:
                st.error(f"Logic Error: {e}")

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please use the form above to Login or Register a new account')