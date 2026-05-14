import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 1. دوال توليد الإشارات (5 أنواع)
# ============================================================

def generate_am(fs=5000):
    t = np.linspace(0, 1, fs, endpoint=False)
    return np.sin(2 * np.pi * 100 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))

def generate_fm(fs=5000):
    t = np.linspace(0, 1, fs, endpoint=False)
    return np.sin(2 * np.pi * (100 * t + 20 * (np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs)))

def generate_fsk(fs=5000):
    t = np.linspace(0, 1, fs, endpoint=False)
    f1, f2 = 50, 150
    data = np.repeat(np.random.randint(0, 2, 10), 500)
    freqs = np.where(data == 0, f1, f2)
    return np.sin(2 * np.pi * freqs * t)

def generate_qam4(fs=5000):
    t = np.linspace(0, 1, fs, endpoint=False)
    phases = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
    phase = np.random.choice(phases)
    return np.sin(2 * np.pi * 100 * t + phase)

def generate_qam16(fs=5000):
    t = np.linspace(0, 1, fs, endpoint=False)
    points = [-3, -1, 1, 3]
    I, Q = np.random.choice(points), np.random.choice(points)
    sig = (I * np.cos(2 * np.pi * 100 * t) - Q * np.sin(2 * np.pi * 100 * t))
    return sig / np.max(np.abs(sig))

def get_spec(signal, fs=5000):
    _, _, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min())

# ============================================================
# 2. تجهيز الـ Dataset
# ============================================================
X, y = [], []
print("Preparing Dataset... 🚧")
samples_per_class = 200

for _ in range(samples_per_class):
    noise = lambda: np.random.normal(0, 0.1, 5000)
    # 0: AM
    X.append(get_spec(generate_am()   + noise())); y.append(0)
    # 1: FM
    X.append(get_spec(generate_fm()   + noise())); y.append(1)
    # 2: FSK
    X.append(get_spec(generate_fsk()  + noise())); y.append(2)
    # 3: 4-QAM
    X.append(get_spec(generate_qam4() + noise())); y.append(3)
    # 4: 16-QAM
    X.append(get_spec(generate_qam16()+ noise())); y.append(4)

X = np.array(X)
y = np.array(y)

print(f"X shape before reshape: {X.shape}")
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
print(f"X shape after reshape:  {X.shape}")  # (1000, 129, 38, 1)

# ============================================================
# 3. تقسيم البيانات
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 4. بناء الـ CNN - 5 classes ✅
# ============================================================
model = models.Sequential([
    layers.Input(shape=(129, 38, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')   # ✅ 5 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 5. التدريب
# ============================================================
print("\nTraining started... 🔥")
history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test)
)

# ============================================================
# 6. حفظ الموديل
# ============================================================
model.save("signal_cnn_model.h5")
print("CNN Model Saved! ✅")

# حفظ نسخة TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite Model Saved! ✅")

# ============================================================
# 7. Confusion Matrix
# ============================================================
print("\nGenerating Confusion Matrix... 📊")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

labels = ['AM', 'FM', 'FSK', '4-QAM', '16-QAM']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Signal Classification - Confusion Matrix 📋')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Done! 🏆")
