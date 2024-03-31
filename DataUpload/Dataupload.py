from flask import *
from fileinput import filename
from flask_restful import Resource, Api
from pymongo import MongoClient
import os
import pandas as pd

app = Flask(__name__)
api = Api(app)

# Initialize MongoDB client
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
image_collection = db['images']  # Collection to store image metadata


class ImageUpload(Resource):
    def post(self):
        if 'file' not in request.files:
            return 'No file part', 400

        f = request.files['file']
        file_name = f.filename

        filepath = os.path.join('/<user_id>/<project_id>/images', file_name)
        if not os.path.exists('/<user_id>/<project_id>/images'):
            os.makedirs('/<user_id>/<project_id>/images')
            filepath = os.path.join('/<user_id>/<project_id>/images', file_name)
        try:
            f.save(filepath)
        except Exception as e:
            return str(e), 500
        user_id = request.form.get('user_id')
        project_id = request.form.get('project_id')
        image_id = request.form.get('image_id')
        # Store image metadata in MongoDB
        image_data = {
            'user_id': user_id,
            'project_id': project_id,
            'image_id': image_id,
            'filename': filepath,
            'label': None
        }
        # insert file name
        try:
            image_collection.insert_one(image_data)
        except Exception as e:
            return str(e), 500

        return 'Image uploaded successfully', 200


class LabelUpload(Resource):
    def post(self):
        if 'file' not in request.files:
            return 'No file part', 400

        f = request.files['file']
        label_filename = f.filename
        try:
            f.save(label_filename)
        except Exception as e:
            return str(e), 500
        user_id = request.form.get('user_id')
        project_id = request.form.get('project_id')
        image_id = request.form.get('image_id')
        # Update image metadata with label information
        image_query = {'user_id': user_id, 'project_id': project_id, 'image_id': image_id}
        try:
            image_collection.update_one(image_query, {'$set': {'label': label_filename}})
        except Exception as e:
            return str(e), 500

        return 'Label uploaded successfully', 200


class ParquetExport(Resource):
    def get(self):
        # Get all documents from the collection
        cursor = image_collection.find({})

        # Create a DataFrame
        df = pd.DataFrame(list(cursor))

        # Save DataFrame to a Parquet file
        df.to_parquet('image_data.parquet')

        # Send file
        return send_file('image_data.parquet', as_attachment=True)


api.add_resource(ImageUpload, '/upload_images')
api.add_resource(LabelUpload, '/upload_label')
api.add_resource(ParquetExport, '/export_to_parquet')
if __name__ == '__main__':
    app.run(debug=True)
