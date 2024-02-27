from flask import Flask, request
import os
import pandas as pd

app = Flask(__name__)

@app.route('/upload_images/<user_id>/<project_id>', methods=['POST'])
def upload_images(user_id, project_id):
    if 'file' not in request.files:
        return 'No file part', 400
    f = request.files['file']
    f.save(f.filename)
    return 'Image uploaded successfully', 200

@app.route('/upload_label/<user_id>/<project_id>', methods=['POST'])
def upload_label(user_id, project_id):
    if 'file' not in request.files:
        return 'No file part', 400
    f = request.files['file']
    f.save(f.filename)
    return 'Label uploaded successfully', 200

if __name__ == '__main__':
    app.run(debug=True)
