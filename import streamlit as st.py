import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

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

    # إرسال للـ API
    with st.spinner('Analyzing signal...'):
        try:
            url = "http://127.0.0.1:8000/predict"
            response = requests.post(url, json={"data": signal.tolist()})
            res = response.json()
            
            # عرض النتيجة بانبهار
            st.success(f"### Prediction: {res['prediction']}")
            st.info(f"### Confidence: {res['confidence']}")
        except:
            st.error("Make sure your FastAPI server is running!")
            #app2.py اعمله حفظ كده