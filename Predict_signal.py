import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram

# 1. تحميل الموديل اللي إنت لسه صانعه
model = tf.keras.models.load_model("signal_cnn_model.h5")

# 2. دالة لتوليد إشارة "مجهولة" للاختبار
def generate_unknown_signal():
    fs = 5000
    t = np.linspace(0, 1, fs, endpoint=False)
    # هنولد إشارة FM مثلاً ونضيف عليها شوشرة جامدة
    signal = np.sin(2 * np.pi * (100 * t + 20 * (np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs)))
    noisy_signal = signal + np.random.normal(0, 0.1, len(signal)) # Noise عالي
    return noisy_signal

# 3. دالة تحويل الإشارة لـ Spectrogram (نفس اللي استخدمناها في التدريب)
def prepare_for_prediction(signal):
    _, _, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())
    # لازم يكون الشكل (1, 129, 38, 1) عشان الموديل يقبله
    return Sxx_norm.reshape(1, 129, 38, 1)

# 4. تنفيذ التوقع
test_signal = generate_unknown_signal()
processed_signal = prepare_for_prediction(test_signal)

prediction = model.predict(processed_signal)
class_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

# 5. النتيجة النهائية
classes = ["AM Signal 📡", "FM Signal 📻"]
print(f"\n--- Prediction Result ---")
print(f"Detected Type: {classes[class_index]}")
print(f"Confidence: {confidence:.2f}% 🔥")