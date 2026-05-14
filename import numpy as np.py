# Beginning Day 1
import numpy as np

# Generate time axis
t = np.linspace(0, 1, 1000)

# Generate signal (simple sine wave)
signal = np.sin(2 * np.pi * 5 * t)

print("Signal generated 🔥")
# Extract features
mean = np.mean(signal)
variance = np.var(signal)
maximum = np.max(signal)
minimum = np.min(signal)
power = np.mean(signal**2)

print("Mean:", mean)
print("Variance:", variance)
print("Max:", maximum)
print("Min:", minimum)
print("Power:", power)
# Add noise
noise = np.random.normal(0, 0.2, len(signal))
noisy_signal = signal + noise

print("Noise added 🔥")
mean_noisy = np.mean(noisy_signal)
print("Noisy Mean:", mean_noisy)
#end of Day 1
 #######################
# Beginning Day 2
def generate_am():
    t = np.linspace(0, 1, 1000)
    
    carrier = np.sin(2 * np.pi * 50 * t)   # high freq
    mod = 1 + 0.5 * np.sin(2 * np.pi * 5 * t)  # low freq
    
    return carrier * mod

def generate_fm():
    t = np.linspace(0, 1, 1000)
    
    return np.sin(2 * np.pi * (50 + 10*np.sin(2*np.pi*5*t)) * t)

am_signal = generate_am()
fm_signal = generate_fm()

print("AM & FM generated 🔥")
def add_noise(signal):
    noise = np.random.normal(0, 0.2, len(signal))
    return signal + noise

am_signal = add_noise(am_signal)
fm_signal = add_noise(fm_signal)
def extract_features(signal):
    return [
        np.mean(signal),
        np.var(signal),
        np.max(signal),
        np.min(signal),
        np.mean(signal**2)
    ]
X = []
y = []

for _ in range(100):
    am = add_noise(generate_am())
    fm = add_noise(generate_fm())

    X.append(extract_features(am))
    y.append("AM")

    X.append(extract_features(fm))
    y.append("FM")

print("Dataset created 🔥", len(X))
#end of Day 2
######################
# Beginning Day 3
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

print("Model trained 🔥")
test_signal = add_noise(generate_am())

features = extract_features(test_signal)

prediction = model.predict([features])

print("Prediction:", prediction[0])

for _ in range(5):
    test_signal = add_noise(generate_fm())
    features = extract_features(test_signal)
    prediction = model.predict([features])
    print("Predicted:", prediction[0])
    from sklearn.metrics import accuracy_score

preds = model.predict(X)
acc = accuracy_score(y, preds)

print("Accuracy:", acc)
# end Day 3
######################
# Beginning Day 4
# 1. توليد إشارة اختبار جديدة (FM مثلاً)
test_signal = add_noise(generate_fm())
features = extract_features(test_signal)

# 2. التوقع (النوع)
prediction = model.predict([features])

# 3. حساب نسبة التأكد (Probability)
probabilities = model.predict_proba([features])
confidence = np.max(probabilities) * 100

print(f"--- Final Result ---")
print(f"Signal Type: {prediction[0]}")
print(f"Confidence Score: {confidence:.2f}% 🔥")
import joblib

# حفظ الموديل في ملف
joblib.dump(model, "model.pkl")

print("Model saved to model.pkl 🔥✅")
# تحميل الموديل من الملف
#loaded_model = joblib.load("model.pkl")

# توليد إشارة جديدة تماماً (خارج بيانات التدريب)
#test_signal = add_noise(generate_fm()) # جرب مرة FM ومرة AM
#features = extract_features(test_signal)

# التوقع باستخدام الموديل المحفوظ
#final_prediction = loaded_model.predict([features])

#print(f"The System says this signal is: {final_predi#ction[0]} 🎯")