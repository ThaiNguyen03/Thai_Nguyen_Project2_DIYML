from flask import Flask, request
from flask_restful import Resource, Api
from pymongo import MongoClient
import os

app = Flask(__name__)
api = Api(app)

# Initialize MongoDB client
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
image_collection = db['images']  # Collection to store image metadata

class ImageUpload(Resource):
    def post(self, user_id, project_id):
        if 'file' not in request.files:
            return 'No file part', 400

        f = request.files['file']
        filename = f.filename
        f.save(filename)
        filepath = os.path.join('/path/to/save', filename)
        f.save(filepath)

        # Store image metadata in MongoDB
        image_data = {
            'user_id': user_id,
            'project_id': project_id,
            'filename': filepath,
            'label': None
        }
        # insert file name
        image_collection.insert_one(image_data)

        return 'Image uploaded successfully', 200

class LabelUpload(Resource):
    def post(self, user_id, project_id):
        if 'file' not in request.files:
            return 'No file part', 400

        f = request.files['file']
        label_filename = f.filename
        f.save(label_filename)

        # Update image metadata with label information
        image_query = {'user_id': user_id, 'project_id': project_id}
        image_collection.update_one(image_query, {'$set': {'label': label_filename}})

        return 'Label uploaded successfully', 200

api.add_resource(ImageUpload, '/upload_images/<user_id>/<project_id>')
api.add_resource(LabelUpload, '/upload_label/<user_id>/<project_id>')

if __name__ == '__main__':
    app.run(debug=True)
