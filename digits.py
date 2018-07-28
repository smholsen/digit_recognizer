from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import base64
from random import *
import time
from models.cnn.cnn_model import Model

app = Flask(__name__,
            static_folder="./dist/static",
            template_folder="./dist")
CORS(app)

model = Model()


@app.route('/api/predict', methods=["POST"])
def predict():
    # Predict digit based on drawing
    # Extract b64 img (remove prefix data:image/png;base64,)
    img = request.values['data_url_base_64'].split(',')[1]
    img = base64.decodebytes(img.encode('utf-8'))
    prediction = model.predict(img)

    return jsonify({'prediction': prediction})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0')
