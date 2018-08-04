from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import base64
from models.cnn.cnn_model import Model
import os

app = Flask(__name__,
            static_folder="./dist/static",
            template_folder="./dist")

# if 'APP_DEBUG' in os.environ:
#     app.debug = True
#     from werkzeug.debug import DebuggedApplication
#     app.wsgi_app = DebuggedApplication(app.wsgi_app, True)

CORS(app)

model = Model()


@app.route('/api/predict', methods=["POST"])
def predict():
    # Predict digit based on drawing
    # Extract b64 img (remove prefix data:image/png;base64,)
    img = request.get_json()['data_url_base_64'].split(',')[1]
    img = base64.decodebytes(img.encode('utf-8'))
    prediction = model.predict(img)

    return jsonify({'prediction': str(prediction)})


@app.route('/api/train', methods=["POST"])
def train():
    # Predict digit based on drawing
    # Extract b64 img (remove prefix data:image/png;base64,)
    img = request.get_json()['data_url_base_64'].split(',')[1]
    img = base64.decodebytes(img.encode('utf-8'))
    label = request.get_json()['label']
    model.train_on_single_img(img, label)

    return jsonify({'trained': "OK"})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0')
