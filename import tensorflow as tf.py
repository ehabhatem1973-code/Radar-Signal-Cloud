import tensorflow as tf
from  tensorflow import keras
# تحميل الموديل 
model = keras.models.load_model('signal_cnn_model.h5')
# عرض معلومات النموذج 
model.summary()
