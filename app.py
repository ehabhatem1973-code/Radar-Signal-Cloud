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
# تأكد من وضع رابط الشيت في الـ Secrets كما في الصورة
conn = st.connection("gsheets", type=GSheetsConnection)

def get_all_users():
    # قراءة البيانات من الشيت (الأعمدة: Name, User Name, Password)
    df = conn.read(ttl=0) 
    creds = {"usernames": {}}
    for _, row in df.iterrows():
        u_name = str(row['User Name'])
        creds["usernames"][u_name] = {
            "name": str(row['Name']),
            "password": str(row['Password'])
        }
    return creds

# تحميل اليوزرز من الشيت عند بداية التشغيل
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

# --- 2. التحكم في واجهة المستخدم (Login/Register) ---
login_placeholder = st.empty()

# إذا لم يكن المستخدم مسجلاً دخوله بعد
if not st.session_state.get('authentication_status'):
    with login_placeholder.container():
        # نموذج التسجيل (Sign Up)
        try:
            # استخدام 'main' لظهورها في منتصف الصفحة
            if authenticator.register_user(location='main'):
                all_users = st.session_state['config']['credentials']['usernames']
                
                # تحويل البيانات لـ DataFrame ورفعها للشيت
                new_df = pd.DataFrame.from_dict(all_users, orient='index').reset_index()
                new_df.columns = ['User Name', 'Name', 'Password']
                conn.update(data=new_df)
                
                # --- حركة الدخول التلقائي بعد التسجيل ---
                new_username = list(all_users.keys())[-1]
                st.session_state['authentication_status'] = True
                st.session_state['username'] = new_username
                st.session_state['name'] = all_users[new_username]['name']
                
                st.success('Registration successful! Redirecting...')
                st.rerun() # إعادة تحميل لفتح صفحة الرادار فوراً
        except Exception as e:
            st.error(f"Sign-up Error: {e}")
        
        # نموذج الدخول التقليدي
        result = authenticator.login(location='main')

# فك تشفير النتيجة للنسخة 0.4.2
if isinstance(st.session_state.get('authentication_status'), bool):
    authentication_status = st.session_state['authentication_status']
    name = st.session_state.get('name')
    username = st.session_state.get('username')
else:
    authentication_status = None

# --- 3. صفحة الرادار (تظهر فقط بعد نجاح الدخول أو التسجيل) ---
if authentication_status:
    login_placeholder.empty() # مسح شاشة الدخول تماماً
    
    # واجهة الخروج في الجنب
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f'Welcome Engineer *{name}*')

    # تسجيل الدخول في ملف Log محلي
    with open("logins.txt", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"User: {name} ({username}) | Login Time: {now}\n")

    # محتوى الصفحة الرئيسي
    st.title("📡 Radar Signal Intelligence")
    st.markdown("---")
    st.write("Secure Cloud Access: **Granted**")

    # دالة الـ Spectrogram
    def get_spec(signal):
        _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

    # تحميل الموديل الذكي
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model('signal_cnn_model.h5')

    model = load_my_model()

    signal_option = st.selectbox("Select Signal Type to Generate:", ["AM Signal", "FM Signal"])

    if st.button("Generate & Classify 🚀"):
        fs = 5000
        t = np.linspace(0, 1, fs, endpoint=False)
        
        if signal_option == "AM Signal":
            signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
        else:
            signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
        
        # 1. رسم الموجة الزمنية
        st.subheader("1. Time Domain Representation")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(t[:500], signal[:500], color='dodgerblue')
        ax1.set_title(f"Time Domain: {signal_option}")
        st.pyplot(fig1)

        with st.spinner('AI analyzing signal fingerprint...'):
            try:
                spec = get_spec(signal)
                input_data = spec.reshape(1, 129, 38, 1)
                prediction = model.predict(input_data)
                classes = ['AM', 'FM']
                res = classes[np.argmax(prediction)]
                conf = np.max(prediction) * 100

                # 2. رسم الـ Spectrogram
                st.subheader("2. Spectrogram (Signal Fingerprint)")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                ax2.set_ylabel("Frequency")
                ax2.set_xlabel("Time")
                st.pyplot(fig2)


                # 3. عرض نتائج التحليل
                st.subheader("3. Intelligence Analysis")
                c1, c2 = st.columns(2)
                c1.metric("Predicted Class", res)
                c2.metric("Confidence Level", f"{conf:.2f}%")

                

            except Exception as e:
                st.error(f"Analysis Error: {e}")

elif authentication_status == False:
    st.error('Username/password is incorrect')