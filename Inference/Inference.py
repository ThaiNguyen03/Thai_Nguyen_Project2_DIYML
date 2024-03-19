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
        model_name = request.json.get('model_name')
        image_file = request.files['image']


        # Load the model and tokenizer
        my_model = AutoModelForImageClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(image_file, return_tensors="pt")

        # Run inference
        classifier = pipeline("image-classification", model= my_model)
        outputs = my_model(**inputs)

        # Get the predicted class
        predicted_label_idx = outputs.logits.argmax(-1).item()

        return {"message": "Inference run successfully", "predicted_class_idx": model.config.id2label[predicted_label_idx]}, 200

api.add_resource(InferenceAPI, '/inference')

if __name__ == '__main__':
    app.run(debug=True)
