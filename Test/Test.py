from flask import Flask, request
from flask_restful import Resource, Api
from PIL import Image
from transformers import AutoModelForImageClassification, AutoTokenizer, AutoImageProcessor
from transformers import pipeline
import os
import torch
from datasets import load_dataset

app = Flask(__name__)
api = Api(app)


class TestAPI(Resource):
    def post(self):
        data = request.get_json()
        model_path = data.get('model_path')
        dataset_path = data.get('dataset_path')
        model_name = data.get('model_name')

        # Load the dataset
        try:
            api_dataset = load_dataset('parquet', data_files=dataset_path)
        except Exception as e:
            return {"message": "dataset not found"}, 404

        results = []

        for i in range(len(api_dataset["train"])):
            try:
                pil_image = api_dataset["train"][i]["image"]
            except Exception as e:
                return {"message": "Image not found"}, 404
            image_processor = AutoImageProcessor.from_pretrained(model_path)
            inputs = image_processor(pil_image, return_tensors="pt")

            my_model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
            with torch.no_grad():
                logits = my_model(**inputs).logits
            predicted = logits.argmax(-1).item()

            results.append(my_model.config.id2label[predicted])

        return {"message": "Inference run successfully",
                "results": results
                }, 200


api.add_resource(TestAPI, '/test')

if __name__ == '__main__':
    app.run(debug=True)
