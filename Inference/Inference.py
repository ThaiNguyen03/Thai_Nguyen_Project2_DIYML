from flask import Flask, request
from flask_restful import Resource, Api
from PIL import Image
from transformers import AutoModelForImageClassification, AutoTokenizer, AutoImageProcessor
from transformers import pipeline
import os
import torch

app = Flask(__name__)
api = Api(app)

class InferenceAPI(Resource):
    def post(self, user_id, project_id):
        data = request.get_json()
        model_path = data.get('model_path')
        image_path = data.get('image_path')
        model_name = data.get('model_name')
        try:
            pil_image = Image.open(image_path)
        except Exception as e:
            return {"message":"Image not found"}, 404
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        inputs = image_processor(pil_image, return_tensors = "pt")

        my_model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
        with torch.no_grad():
            logits = my_model(**inputs).logits
        predicted = logits.argmax(-1).item()

        return {"message": "Inference run successfully",
                "results": my_model.config.id2label[predicted]
                }, 200

api.add_resource(InferenceAPI, '/inference/<string:user_id>/<string:project_id>')

if __name__ == '__main__':
    app.run(debug=True)
