import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # ضفنا ده وشلنا requests

# تحميل الموديل مرة واحدة في أول تشغيل البرنامج
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('signal_cnn_model.h5')

model = load_my_model()

st.set_page_config(page_title="Signal Intelligence Radar", page_icon="📡")

st.title("📡 Signal Classification Radar")
st.write("Welcome, Engineer! Upload or Generate a signal to classify it using AI.")

# 1. قائمة اختيار نوع الإشارة
signal_type = st.selectbox("Select Signal Type to Generate:", ["AM Signal", "FM Signal"])

if st.button("Generate & Classify 🚀"):
    # توليد إشارة حقيقية (5000 عينة) عشان الموديل ينبهر
    fs = 5000
    t = np.linspace(0, 1, fs, endpoint=False)
    
    if signal_type == "AM Signal":
        signal = (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)
    else:
        signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
    
    # رسم الإشارة
    fig, ax = plt.subplots()
    ax.plot(t[:500], signal[:500]) # رسم أول 500 عينة بس للوضوح
    ax.set_title(f"Generated {signal_type} (First 500 samples)")
    st.pyplot(fig)

with st.spinner('Analyzing signal...'):
            try:
                import numpy as np
                
                # 1. الحجم المستهدف بناءً على رسالة الخطأ
                # 15360 هو الرقم السحري اللي الموديل عايزه
                target_size = 15360 
                
                # 2. معالجة الإشارة لتناسب الحجم
                if len(signal) > target_size:
                    processed_signal = signal[:target_size]
                else:
                    processed_signal = np.pad(signal, (0, target_size - len(signal)))

                # 3. التعديل الجوهري للأبعاد (Reshape)
                # هنخليها 120 في 128 عشان المجموع يبقى 15360
                input_data = processed_signal.reshape(1, 120, 128, 1)
                
                # 4. التوقع
                prediction = model.predict(input_data)
                
                classes = ['AM', 'FM'] 
                res = classes[np.argmax(prediction)]
                conf = np.max(prediction) * 100

                st.success(f"### Prediction: {res}")
                st.info(f"### Confidence: {conf:.2f}%")

            except Exception as e:
                st.error(f"Error logic: {e}")
            #import requests هشيل ديه عشان محتاجش اعمل سرفر تاني 