from flask import Flask, render_template,request
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import torch

app = Flask(__name__)

model1 = tf.keras.models.load_model('mobilenet-main.h5')

# model2 = tf.keras.models.load_model('custom1.hdf5')

# model3 = torch.load('vgg16.h5')


class_labels = ['drinking',
 'hair and makeup',
 'normal driving',
 'operating the radio',
 'reaching behind',
 'talking on the phone - left',
 'talking on the phone - right',
 'talking to passenger',
 'texting - left',
 'texting - right']

d ={0:'drinking',
 1:'hair and makeup',
 2:'normal driving',
 3:'operating the radio',
 4:'reaching behind',
 5:'talking on the phone - left',
 6:'talking on the phone - right',
 7:'talking to passenger',
 8:'texting - left',
 9:'texting - right'}
 
@app.route('/')
def home():
    return render_template('index.html')

 
@app.route('/predict', methods=['POST'])
def predict():
    
    img_file = request.files['file']
    img_array = np.frombuffer(img_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    mn_class_label=make_prediction(img,model1)
    
    # custom_class_label=make_prediction(img,model2)

    # vgg_class_label=make_prediction(img,model3)
    
    return render_template('prediction.html', class_label1=mn_class_label)
def make_prediction(img,model):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return d[predicted_class]


 
if __name__ == '__main__':
    app.run(debug=True)

