from flask import Flask, request
from flask_restful import Resource, Api
from PIL import Image
from transformers import AutoModelForImageClassification, AutoTokenizer
from transformers import pipeline
import os

app = Flask(__name__)
api = Api(app)

class InferenceAPI(Resource):
    def post(self):
        data = request.get_json()
        model_path = data.get('model_path')
        image_path = data.get('image_path')
        model_name = data.get('model_name')



        my_model = AutoModelForImageClassification.from_pretrained(model_path)
        user_classifier= pipeline("image-classification",model = my_model)
        results = user_classifier(image_path)

        return {"message": "Inference run successfully",
                "results": results
                }, 200

api.add_resource(InferenceAPI, '/inference')

if __name__ == '__main__':
    app.run(debug=True)
