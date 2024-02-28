from flask import *
from pymongo import MongoClient

app = Flask(__name__)

# Initialize MongoDB client
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
image_collection = db['images']  # Collection to store image metadata

@app.route('/upload_images/<user_id>/<project_id>', methods=['POST'])
def upload_images(user_id, project_id):
    if 'file' not in request.files:
        return 'No file part', 400

    f = request.files['file']
    filename = f.filename
    f.save(filename)

    # Store image metadata in MongoDB
    image_data = {
        'user_id': user_id,
        'project_id': project_id,
        'filename': filename,
        'label': None
    }
    image_collection.insert_one(image_data)

    return 'Image uploaded successfully', 200

@app.route('/upload_label/<user_id>/<project_id>', methods=['PATCH'])
def upload_label(user_id, project_id, label_value):
    if 'file' not in request.files:
        return 'No file part', 400

    f = request.files['file']
    label_filename = f.filename
    f.save(label_filename)

    # Update image metadata with label information
    image_query = {'user_id': user_id, 'project_id': project_id}
    image_collection.update_one(image_query, {'$set': {'label': label_filename}})

    return 'Label uploaded successfully', 200

if __name__ == '__main__':
    app.run(debug=True)
