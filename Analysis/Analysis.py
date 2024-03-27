from flask import Flask, request
from flask_restful import Resource, Api
from PIL import Image
import numpy as np
import io
import pymongo
from pymongo import MongoClient
import base64
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize
from datasets import load_dataset

app = Flask(__name__)
api = Api(app)

# Setup MongoDB connection
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
collection = db['Analysis_data']


class Analysis(Resource):
    def post(self):
        # Load the data
        user_data = request.get_json()
        dataset_path = user_data.get('dataset_path')
        model_name = user_data.get('model_name')

        api_dataset = load_dataset('parquet', data_files=dataset_path)

        sizes = []
        means = []
        std_devs = []

        checkpoint = model_name
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )
        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

        # Analyze each image
        for i in range(len(api_dataset["train"])):

            image = _transforms(api_dataset["train"][i]["image"])

            size = image.size()
            mean = image.mean().item()
            std_dev = image.std().item()

            doc = {'size': size, 'mean': mean, 'std_dev': std_dev, 'image_id': i}
            try:
                collection.insert_one(doc)
            except Exception as e:
                return {'message': 'analysis insertion not successful'}, 400

        # Return the results
        return {'message': 'Image analysis completed and stored in MongoDB.'}, 200

    def get(self):
        user_data = request.get_json()
        image_id = user_data.get('id')

        doc = collection.find_one({'image_id': int(image_id)})

        if doc is None:
            return {'message': 'No image found with the given id.'}, 404

        return {'size': doc['size'], 'mean': doc['mean'], 'std_dev': doc['std_dev']}, 200

api.add_resource(Analysis, '/analysis')

if __name__ == '__main__':
    app.run(debug=True)
