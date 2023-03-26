import subprocess
from flask import Flask, jsonify, make_response
from flask import request
import numpy as np
from model import ONNXModel, ImagePreprocessor
import os
import json
model = ONNXModel()
preprocessor = ImagePreprocessor()

# Create the http server app
server = Flask(__name__)
server.config['DEBUG'] = True
server.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
server.config["IMAGE_UPLOADS"] = "images_test/"
server.config["MODEL"] = "models/"


# Healthchecks verify that the environment is correct on Banana Serverless
@server.route('/healthcheck', methods=["GET"])
def healthcheck():
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True
    return {"state": "healthy", "gpu": gpu}, 200


@server.route('/', methods=["POST"]) 
def inference():
    # Check if the POST request has a file part
    if 'image' not in request.files:
        return {
            'success': False,
            'error': 'No file uploaded',
        }, 400

    # Read the file from the request
    image_file = request.files['image']
    if not image_file or image_file.filename == '':
        return {
            'success': False,
            'error': 'No image selected for uploading',
        }, 400
    image_path = os.path.join(server.config["IMAGE_UPLOADS"], image_file.filename)
    image_file.save(image_path)
    input_data = preprocessor.preprocess_numpy(image_path)
    prediction = model.predict(input_data.numpy())
    class_probs = np.exp(prediction)
    predicted_class_id = np.argmax(class_probs)
    return {
            'success': True,
            'predicted_class_id': str(predicted_class_id),
        }, 200


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8000)
