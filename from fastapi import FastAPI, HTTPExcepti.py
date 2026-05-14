from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram
from typing import List

# 1. تعريف التطبيق وتحميل الموديل
app = FastAPI(title="Signal Classifier API 📡")
model = tf.keras.models.load_model(r"C:\me\My cources\Huawie course\Project\signal_cnn_model.h5")

# 2. هيكل البيانات المتوقع (JSON)
class SignalData(BaseModel):
    data: List[float]

# 3. دالة المعالجة (نفس اللي استخدمناها في التدريب)
# ابحث عن الدالة دي في main.py وعدلها كدة:
def process_signal(signal_list):
    signal = np.array(signal_list)
    
    # بدل ما نرفض الإشارة، هنخليها دايماً 5000 رقم
    # لو قصيرة هيكملها أصفار (Padding)، ولو طويلة هيقصها
    if len(signal) < 5000:
        signal = np.pad(signal, (0, 5000 - len(signal)), 'constant')
    else:
        signal = signal[:5000]
        
    _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())
    
    return Sxx_norm.reshape(1, 129, 38, 1)

# 4. نقطة النهاية (Endpoint) للتوقع
@app.post("/predict")
async def predict_signal(input_data: SignalData):
    try:
        # معالجة الإشارة
        processed = process_signal(input_data.data)
        
        # التوقع
        prediction = model.predict(processed)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        classes = ["AM", "FM"]
        
        return {
            "prediction": classes[class_idx],
            "confidence": f"{confidence * 100:.2f}%",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# لتشغيل السيرفر: uvicorn main:app --reload
#signal_cnn_model.h5 ده فولدر شبه الاساسي عشان هعدل فالاساسي