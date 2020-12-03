from flask import Flask, request, redirect, url_for, render_template
from markupsafe import escape
from joblib import load
import os
from werkzeug.utils import secure_filename
import json

from keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import Session, get_default_graph

import numpy as np
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

global sess
sess = Session()
set_session(sess)

global base_model
base_model = load_model('base_model.h5')

global classifier
classifier = load('model_150.joblib')

global graph
graph = get_default_graph()

global names_top_artists
names_top_artists = load('names_top_artists.joblib')

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('static', filename))
        return redirect(url_for('predict_result', filename = filename))

    else: return render_template('index.html')

@app.route('/prediction/<filename>')

def predict_result(filename):
    path = os.path.join('static', filename)
    html_path = "/static/" + filename.replace("\\", "")
    image = load_img(path, target_size=(224, 224)) # returns a PIL image
    image = img_to_array(image) # turn into array
    # reshape into correct format for the model (samples x H x W x channels)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # extract features
    features = base_model.predict(image)
    # predict the artist
    yhat = classifier.predict(features.reshape((1, 7*7*512)))
    prediction = names_top_artists[int(yhat)].replace("_", " ") # name of the artist

    return render_template('predictions.html', prediction=prediction, image_path=html_path)

@app.context_processor
def artists_info():
    artists_info_dict = json.load(open(".\\static\\artists_info.json"))
    return dict(artists_info_dict=artists_info_dict)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=5000)
