import requests
import numpy as np

# 1. توليد إشارة FM تجريبية (5000 عينة)
fs = 5000
t = np.linspace(0, 1, fs, endpoint=False)
# إشارة FM حقيقية
signal = np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))
signal_list = signal.tolist()

# 2. إرسال الإشارة للـ API
url = "http://127.0.0.1:8000/predict"
payload = {"data": signal_list}

print("Sending signal to Radar API... 📡")
response = requests.post(url, json=payload)

# 3. عرض النتيجة
if response.status_code == 200:
    result = response.json()
    print(f"✅ Prediction: {result['prediction']}")
    print(f"🔥 Confidence: {result['confidence']}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)