import os
from flask import Flask, request, redirect, url_for, jsonify, Response, render_template
from werkzeug.utils import secure_filename
import pickle
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import keras
app = Flask(__name__)
model = None
classes = ["ApplesGradeA", "ApplesGradeB", "ApplesGradeC", "BananaGradeA", "BananaGradeA", "BananaGradeC", "LimeGradeA", "LimeGradeB", "LimeGradeC",
           "OrangesGradeA", "OrangesGradeB", "OrangesGradeC", "PomegranateGradeA", "PomegranateGradeB", "PomegranateGradeC"]
def load_model():
    global model
    model = keras.models.load_model("D:/Thesis/Models/vgg/saved_vggs/saved_vggs/vgg_model11")

def preprocess(img):
    if  img.mode != 'RGB':
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        inputs = preprocess_input(img)
    return inputs
@app.route('/')
def index_page():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    global classes
    response = {'success': False}
    if request.method == 'POST':
        if request.files.get('file'): # image is stored as name "file"
            img_requested = request.files['file'].read()
            img = Image.open(io.BytesIO(img_requested))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            inputs = preprocess_input(img)

            preds = model.predict(inputs)

            print("preds", preds)
            index = np.argmax(preds, axis=1)
            print(index[0])
            print(classes[index[0]])
            response['predictions'] = ""
            #row = {'label': str(classes[index[0]]), 'probability': str(preds[0][index[0]])} # numpy float is not good for json
            response['predictions'] = str(classes[index[0]])
            response['success'] = True
            return jsonify(response)

    return render_template('upload.html')

@app.route('/probabilities', methods=['GET', 'POST'])
def get_probabilities():
    global classes
    response = {'success': False}
    if request.method == 'POST':
        if request.files.get('file'): # image is stored as name "file"
            img_requested = request.files['file'].read()
            img = Image.open(io.BytesIO(img_requested))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            inputs = preprocess_input(img)

            preds = model.predict(inputs)

            print("preds", preds)
            index = np.argmax(preds, axis=1)
            print(index[0])
            print(classes[index[0]])
            response['predictions'] = []
            row = {
                   str(classes[0]): str(preds[0][0]),
                   str(classes[1]): str(preds[0][1]),
                   str(classes[2]): str(preds[0][2]),
                   str(classes[3]): str(preds[0][3]),
                   str(classes[4]): str(preds[0][4]),
                   str(classes[5]): str(preds[0][5]),
                   str(classes[6]): str(preds[0][6]),
                   str(classes[7]): str(preds[0][7]),
                   str(classes[8]): str(preds[0][8]),
                   str(classes[9]): str(preds[0][9]),
                   str(classes[10]): str(preds[0][10]),
                   str(classes[11]): str(preds[0][11]),
                   str(classes[12]): str(preds[0][12]),
                   str(classes[13]): str(preds[0][13]),
                   str(classes[14]): str(preds[0][14]),
                   }
            response['predictions'].append(row)
            response['success'] = True
            return jsonify(response)

    return render_template('upload.html')
if __name__ == '__main__':
    load_model()
    # no-thread: https://github.com/keras-team/keras/issues/2397#issuecomment-377914683
    # avoid model.predict runs before model initiated
    # To let this run on HEROKU, model.predict should run onece after initialized
    app.run(threaded=False)