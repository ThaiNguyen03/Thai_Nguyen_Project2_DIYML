from flask import *
from distutils.log import debug
from fileinput import filename
import os
import pandas as pd
app = Flask(__name__)
@app.route("/")
@app.route('/upload_images', methods=['POST'])
def upload_images(user_id, project_id):
    f = request.files['file']
    f.save(f.filename)

    return 'Image uploaded successfully'

@app.route('/upload_label/', methods=['POST'])
def upload_label(user_id, project_id):
    f = request.files['file']
    f.save(f.filename)
    return 'Label uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)


