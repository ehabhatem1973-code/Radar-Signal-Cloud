import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
import streamlit as st
import streamlit_authenticator as stauth
from datetime import datetime 
 # عشان نسجل الوقت بالظبط
# ... (كود التعريفات والباسوردات اللي قلناه قبل كدة) ...

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # أول ما ينجح في الدخول، هنسجل البيانات دي في ملف
    with open("logins.txt", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"User: {name} ({username}) | Login Time: {now}\n")
    
    authenticator.logout('Logout', 'main')
    st.success(f'Welcome Engineer *{name}*')
    
    
# --- كود الرادار بتاعك بيبدأ من هنا ---
# 1. دالة الـ Spectrogram
def get_spec(signal):
    _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

# 2. تحميل الموديل
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('signal_cnn_model.h5')

model = load_my_model()

# 3. واجهة البرنامج
st.set_page_config(page_title="Radar Intelligence", page_icon="📡")
st.title("📡 Radar Signal Intelligence")
st.write("Welcome, Engineer! Select a signal type and click Generate.")

# اختيار نوع الإشارة (لازم يكون بره الـ IF)
signal_option = st.selectbox("Select Signal Type:", ["AM Signal", "FM Signal"])

if st.button("Generate & Classify 🚀"):
    # توليد الإشارة
    fs = 5000
    t = np.linspace(0, 1, fs, endpoint=False)
    
    if signal_option == "AM Signal":
        signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
    else:
        signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
    
    # الرسم البياني
    fig1, ax1 = plt.subplots()
    ax1.plot(t[:500], signal[:500])
    ax1.set_title(f"Generated {signal_option}")
    st.pyplot(fig1)

    with st.spinner('Analyzing...'):
        try:
            # التحويل لـ Spectrogram
            spec = get_spec(signal)
            input_data = spec.reshape(1, 129, 38, 1)
            
            # التوقع
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