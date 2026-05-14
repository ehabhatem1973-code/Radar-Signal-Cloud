import numpy as np
from scipy.signal import spectrogram

# --- 1. دوال التوليد المعدلة (نفس اللي فاتت) ---
def generate_am(sampling_rate=5000, duration=1.0):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    carrier_freq = 100.0  # تردد عالي
    mod_freq = 5.0      # تردد واطي
    mod_index = 0.5     # مؤشر الـ Modulation
    
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    modulator = np.sin(2 * np.pi * mod_freq * t)
    
    return carrier * (1 + mod_index * modulator)

def generate_fm(sampling_rate=5000, duration=1.0):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    carrier_freq = 100.0
    mod_freq = 5.0
    mod_deviation = 20.0  # مقدار الانحراف الترددي
    
    # تردد الـ FM هو تكامل تردد الـ Modulator
    return np.sin(2 * np.pi * (carrier_freq * t + mod_deviation * (np.cumsum(np.sin(2 * np.pi * mod_freq * t)) / sampling_rate)))

# --- 2. دالة تحويل الـ Spectrogram للـ Dataset (مهمة جداً) ---
# الدالة دي مش بس بترسم، دي بترجع الـ data بتاعت الرسمة
def get_spectrogram_data(signal, sampling_rate=5000):
    # نستخدم نفس الـ parameters عشان الدقة تبقى واحدة
    _, _, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=256, noverlap=128)
    
    # نستخدم الـ Log لتحسين الـ range
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    
    # نعمل normalization عشان الـ data تبقى بين 0 و 1 (مهم جداً للـ CNN)
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())
    
    # نرجع فقط الـ data بتاعت الرسمة
    return Sxx_norm

# --- 3. تكوين الـ Dataset (The Generator Loop) ---
X = [] # الصور (الـ Spectrograms)
y = [] # الـ labels (0 للـ AM و 1 للـ FM)

# عدد الإشارات اللي هنولدها من كل نوع
num_samples_per_class = 200

# الـ Sampling Rate
fs = 5000

print("Generating Dataset... 🚧")

# توليد الـ AM
for _ in range(num_samples_per_class):
    signal = generate_am(sampling_rate=fs)
    # نضيف Noise عشان الموديل يبقى قوي
    noise_power = np.random.uniform(0.05, 0.3)
    noisy_signal = signal + np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    # نحولها لـ Spectrogram data ونضيفها لـ X
    X.append(get_spectrogram_data(noisy_signal, sampling_rate=fs))
    # نضيف الـ label (AM = 0) لـ y
    y.append(0)

# توليد الـ FM
for _ in range(num_samples_per_class):
    signal = generate_fm(sampling_rate=fs)
    noise_power = np.random.uniform(0.05, 0.3)
    noisy_signal = signal + np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    X.append(get_spectrogram_data(noisy_signal, sampling_rate=fs))
    # نضيف الـ label (FM = 1) لـ y
    y.append(1)

# --- 4. تحويل المصفوفات (Arrays) لمصفوفات NumPy النهائية ---
# الـ CNN بيحتاج الصور تبقى (H, W, Channels). بما إن الصور هنا Grayscale، الـ channels بـ 1
X = np.array(X)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
y = np.array(y)

print("Dataset is Ready! 🔥✅")
print("Total Signals Generated:", len(X))
print("Final X shape (Images):", X.shape) # هيطلع مثلاً (400, 129, 75, 1)
print("Final y shape (Labels):", y.shape) # هيطلع (400,)