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
                # 1. تجهيز الإشارة للموديل (Reshape)
                # بنحول المصفوفة لشكل (batch, samples, channels)
                input_signal = signal.reshape(1, -1, 1)
                
                # 2. التوقع المباشر باستخدام الموديل اللي حملناه فوق (model)
                prediction_probs = model.predict(input_signal)
                
                # 3. تحديد النوع الأعلى احتمالية
                # غير الأسماء دي للأنالوغ اللي عندك بالترتيب الصح (AM, FM, etc.)
                classes = ['AM', 'FM'] 
                
                idx = np.argmax(prediction_probs)
                result = classes[idx]
                confidence = np.max(prediction_probs) * 100

                # 4. عرض النتائج فوراً
                st.success(f"### Prediction: {result}")
                st.info(f"### Confidence: {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"حدث خطأ أثناء التصنيف: {e}")
            #import requests هشيل ديه عشان محتاجش اعمل سرفر تاني 