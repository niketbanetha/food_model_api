#!/usr/bin/env python



from flask import Flask,request

from werkzeug.utils import secure_filename


import numpy as np

from PIL import Image

import tensorflow as tf  # TF2

app = Flask(__name__)

def load_labels(filename):
     with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def model_predict(img_path):
    interpreter = tf.lite.Interpreter(
    model_path="model-export_icn_tflite-fruits_20200610010225-2020-06-09T22 59 45.831Z_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(img_path).resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels("model-export_icn_tflite-fruits_20200610010225-2020-06-09T22 59 45.831Z_dict.txt")
    maxresult = []
    for i in top_k:
        if floating_model:

            maxresult.append('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        else:
            maxresult.append('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

    return (maxresult[0].split(": ")[1])


@app.route('/', methods=['GET'])
def index():
    # Main page
    return "Hi this is food model api."


@app.route('/predict', methods=['GET'])
def upload():
        # Make prediction
        if 'id' in request.args:
            id = str(request.args['id'])

        preds = model_predict(id)
        return preds

        #return preds

if __name__ == '__main__':
    app.run(debug=True)



