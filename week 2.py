import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# 1. دوال التوليد (تأكد إن الترددات مختلفة شوية عشان الفرق يبان)
def generate_am():
    t = np.linspace(0, 1, 5000) # زودنا النقط لـ 5000 لدقة أعلى
    carrier = np.sin(2 * np.pi * 100 * t) # تردد عالي 100Hz
    modulator = 1 + 0.5 * np.sin(2 * np.pi * 5 * t)
    return carrier * modulator

def generate_fm():
    t = np.linspace(0, 1, 5000)
    # التردد هنا بيتغير بين 90 و 110Hz
    return np.sin(2 * np.pi * (100 + 20 * np.sin(2 * np.pi * 5 * t)) * t)

# 2. تحويل الـ FM ورسمها (جرب الـ AM بعدها)
signal = generate_fm()

# ركز هنا: زودنا nperseg عشان نعلي الدقة الترددية
f, t, Sxx = spectrogram(signal, fs=5000, nperseg=256, noverlap=128)

# 3. الرسم
plt.figure(figsize=(10, 5))
# استخدمنا log10 عشان نوضح الفروق البسيطة في القوة
plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
plt.ylim(0, 200) # بنركز على منطقة الترددات اللي تهمنا
plt.title('High-Res Spectrogram 🔥')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Intensity [dB]')
plt.show()
