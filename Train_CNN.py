import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from scipy.signal import spectrogram, butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 0. إعدادات عامة
# ============================================================
FS                = 5000
DURATION          = 1
SAMPLES           = FS * DURATION
SAMPLES_PER_CLASS = 300
NOISE_STD         = 0.15

SIGNAL_NAMES = [
    "AM", "FM", "PM",
    "SSB", "DSB",
    "FSK", "ASK", "BPSK",
    "QPSK", "4-QAM", "16-QAM", "64-QAM"
]
NUM_CLASSES = len(SIGNAL_NAMES)   # 12

# ============================================================
# 1. الفلاتر
# ============================================================
def low_pass_filter(signal, cutoff=300, fs=FS, order=5):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

def high_pass_filter(signal, cutoff=20, fs=FS, order=5):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, signal)

def band_pass_filter(signal, low=40, high=200, fs=FS, order=5):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def apply_signal_filter(signal, signal_type):
    signal = high_pass_filter(signal)
    if signal_type in ["FM", "FSK", "PM"]:
        return band_pass_filter(signal, low=40, high=200)
    return low_pass_filter(signal, cutoff=300)

# ============================================================
# 2. توليد الإشارات (12 نوع)
# ============================================================
def generate_am(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    return (1 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.sin(2 * np.pi * 100 * t)

def generate_fm(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    return np.sin(2 * np.pi * (100 * t + 20 * np.cumsum(np.sin(2 * np.pi * 5 * t)) / fs))

def generate_pm(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    m = np.sin(2 * np.pi * 5 * t)
    return np.sin(2 * np.pi * 100 * t + np.pi * m)

def generate_ssb(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    m = np.sin(2 * np.pi * 5 * t)
    carrier = np.cos(2 * np.pi * 100 * t)
    return m * carrier - np.sqrt(1 - m**2 + 1e-8) * np.sin(2 * np.pi * 100 * t)

def generate_dsb(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    m = np.sin(2 * np.pi * 5 * t)
    return m * np.cos(2 * np.pi * 100 * t)

def generate_fsk(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    f1, f2 = 50, 150
    data = np.repeat(np.random.randint(0, 2, 10), 500)
    return np.sin(2 * np.pi * np.where(data == 0, f1, f2) * t)

def generate_ask(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.repeat(np.random.randint(0, 2, 10), 500)
    return np.where(data == 0, 0.1, 1.0) * np.sin(2 * np.pi * 100 * t)

def generate_bpsk(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.repeat(np.random.randint(0, 2, 10), 500)
    return np.sin(2 * np.pi * 100 * t + np.where(data == 0, 0, np.pi))

def generate_qpsk(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    return np.sin(2 * np.pi * 100 * t + np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2]))

def generate_qam4(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    return np.sin(2 * np.pi * 100 * t + np.random.choice([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]))

def generate_qam16(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    pts = [-3, -1, 1, 3]
    I, Q = np.random.choice(pts), np.random.choice(pts)
    sig = I * np.cos(2 * np.pi * 100 * t) - Q * np.sin(2 * np.pi * 100 * t)
    return sig / (np.max(np.abs(sig)) + 1e-8)

def generate_qam64(fs=FS):
    t = np.linspace(0, 1, fs, endpoint=False)
    pts = [-7, -5, -3, -1, 1, 3, 5, 7]
    I, Q = np.random.choice(pts), np.random.choice(pts)
    sig = I * np.cos(2 * np.pi * 100 * t) - Q * np.sin(2 * np.pi * 100 * t)
    return sig / (np.max(np.abs(sig)) + 1e-8)

GENERATORS = [
    generate_am, generate_fm, generate_pm,
    generate_ssb, generate_dsb,
    generate_fsk, generate_ask, generate_bpsk,
    generate_qpsk, generate_qam4, generate_qam16, generate_qam64
]

# ============================================================
# 3. استخراج الـ Parameters
# ============================================================
def estimate_parameters(signal, fs=FS):
    """
    يحسب:
      - Carrier Frequency (Hz)
      - Bandwidth (Hz)
      - Pulse Width (µs)
      - PRI - Pulse Repetition Interval (µs)
    """
    N        = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs    = np.fft.rfftfreq(N, d=1/fs)

    # Carrier Frequency
    carrier_freq = float(freqs[np.argmax(fft_vals)])

    # Bandwidth (-3dB)
    peak_power = np.max(fft_vals**2)
    above      = fft_vals**2 >= (peak_power / 2)
    bw = float(freqs[above][-1] - freqs[above][0]) if np.any(above) else 0.0

    # Pulse Width
    rms    = np.sqrt(np.mean(signal**2))
    active = (np.abs(signal) > rms).astype(int)
    diff   = np.diff(active)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    if len(starts) > 0 and len(ends) > 0:
        if len(ends) > 0 and ends[0] < starts[0]:
            ends = ends[1:]
        n = min(len(starts), len(ends))
        pulse_width = float(np.mean((ends[:n] - starts[:n]) / fs * 1e6))
    else:
        pulse_width = 0.0

    # PRI
    pri = float(np.mean(np.diff(starts)) / fs * 1e6) if len(starts) > 1 else 0.0

    return {
        "Carrier Frequency (Hz)" : round(carrier_freq, 2),
        "Bandwidth (Hz)"         : round(bw, 2),
        "Pulse Width (µs)"       : round(pulse_width, 2),
        "PRI (µs)"               : round(pri, 2),
    }

# ============================================================
# 4. Spectrogram
# ============================================================
def get_spec(signal, fs=FS):
    _, _, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)
    Sxx_log   = 10 * np.log10(Sxx + 1e-10)
    return (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-8)

# ============================================================
# 5. بناء الـ Dataset
# ============================================================
print("📦 Preparing Dataset...")
X, y = [], []

for label, (gen_func, sig_name) in enumerate(zip(GENERATORS, SIGNAL_NAMES)):
    for _ in range(SAMPLES_PER_CLASS):
        noise = np.random.normal(0, NOISE_STD, SAMPLES)
        raw   = gen_func()
        filt  = apply_signal_filter(raw + noise, sig_name)
        X.append(get_spec(filt))
        y.append(label)
    print(f"  ✅ {sig_name} done ({SAMPLES_PER_CLASS} samples)")

X = np.array(X).reshape(-1, 129, 38, 1)
y = np.array(y)
print(f"\n✅ Dataset: {X.shape}  |  {NUM_CLASSES} classes")

# ============================================================
# 6. تقسيم البيانات
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 7. بناء الـ CNN
# ============================================================
def build_model(num_classes=12):
    inp = layers.Input(shape=(129, 38, 1))

    # Block 1
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    # Classifier
    x   = layers.Flatten()(x)
    x   = layers.Dense(512, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return m

model = build_model(NUM_CLASSES)
model.summary()

# ============================================================
# 8. التدريب مع Callbacks
# ============================================================
cb_list = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=12,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=6, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy',
                              save_best_only=True, verbose=1),
]

print("\n🔥 Training started...")
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=cb_list
)

# ============================================================
# 9. حفظ الموديلات
# ============================================================
model.save("signal_cnn_model.h5")
print("✅ signal_cnn_model.h5 saved")

converter    = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ model.tflite saved")

# ============================================================
# 10. Training History Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy'); axes[0].legend()
axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss'); axes[1].legend()
plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# ============================================================
# 11. Confusion Matrix
# ============================================================
print("\n📊 Generating Confusion Matrix...")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm     = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=SIGNAL_NAMES, yticklabels=SIGNAL_NAMES)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title('12-Class Signal Classification — Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=SIGNAL_NAMES))
print("🏆 Done!")
