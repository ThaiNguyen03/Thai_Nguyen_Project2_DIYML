from flask import Flask, request
from flask_restful import Resource, Api
from pymongo import MongoClient
import numpy as np
import sklearn
import evaluate
import torch
import PIL
from datasets import load_from_disk, load_dataset
import json, os
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import Trainer, DefaultDataCollator, TrainingArguments, AutoModelForImageClassification, \
    AutoTokenizer, \
    AutoFeatureExtractor, AutoImageProcessor

app = Flask(__name__)
api = Api(app)

# Initialize MongoDB client
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
model_collection = db['model_data']  # Collection to store model parameters
stats_collection = db['stats']  # Collection to store training stats


class UploadParameters(Resource):
    def post(self):
        data = request.get_json()
        user_id = data.get('user_id')
        project_id = data.get('project_id')
        parameters = data.get('parameters')
        model_collection.insert_one({
            'user_id': user_id,
            'project_id': project_id,
            'parameters': parameters
        })
        return {"message": "Parameters uploaded successfully"}, 200


class StartTraining(Resource):
    def post(self):
        data = request.get_json()
        user_id = data.get('user_id')
        project_id = data.get('project_id')
        model_name = data.get('model_name')
        num_labels = data.get('num_labels')
        labels = data.get('labels')
        train_dataset_path = data.get('train_dataset')
        parameters_data = model_collection.find_one({"user_id": user_id, "project_id": project_id})
        parameters = parameters_data['parameters']
       # api_dataset = load_from_disk(train_dataset_path)
        api_dataset= load_dataset('parquet',data_files='./training_test/training_test.parquet')

        api_dataset = api_dataset["train"].train_test_split(test_size=0.2)
        # Check if 'labels' is in features
       # if 'labels' in api_dataset["train"].features:
        labels = api_dataset["train"].features["label"].names
        #else:
         #   print("The 'labels' feature does not exist in the dataset. Please check the feature names.")
         #   print(api_dataset["train"].features.keys())

        #labels = api_dataset["train"].features["labels"].names
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        checkpoint = model_name
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)

        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )
        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

        def transforms(examples):
            examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
            del examples["image"]
            return examples

        api_dataset = api_dataset.with_transform(transforms)
        model = AutoModelForImageClassification.from_pretrained(model_name,
                                                                num_labels=len(labels),
                                                                id2label=id2label,
                                                                label2id=label2id
                                                                )
        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        # eval_dataset = load_dataset('parquet', data_files='/home/thai/training_test/validation-00000-of-00003.parquet')
        # load training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/{user_id}/{project_id}',
            num_train_epochs=parameters.get('num_train_epochs'),
            evaluation_strategy="epoch",
            save_strategy= "epoch",
            learning_rate=parameters.get('learning_rate'),
            per_device_train_batch_size=parameters.get('per_device_train_batch_size'),
            per_device_eval_batch_size=parameters.get('per_device_eval_batch_size'),
            gradient_accumulation_steps=parameters.get('gradient_accumulation_steps'),
            warmup_ratio=parameters.get('warmup_ratio'),
            logging_steps=parameters.get('logging_steps'),
            load_best_model_at_end=True,
            logging_dir=f'./logs/{user_id}/{project_id}',
            remove_unused_columns=False,
        )
        data_collator = DefaultDataCollator()
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=api_dataset["train"],
            eval_dataset=api_dataset["test"],
            tokenizer=feature_extractor
        )

        trainer.train()

        # Save training stats to the database
        stats_collection.insert_one({
            'user_id': user_id,
            'project_id': project_id,
            'model_name': model_name,
            'training_stats': trainer.evaluate()
        })

        return {"message": f"Training for model {model_name} completed successfully"}, 200


class GetTrainingStats(Resource):
    def get(self):
        data = request.get_json()
        user_id = data.get('user_id')
        project_id = data.get('project_id')
        model_name = data.get('model_name')
        stats = stats_collection.find_one({
            'user_id': user_id,
            'project_id': project_id,
            'model_name': model_name
        })

        if stats:
            return stats['training_stats'], 200
        else:
            return {"message": "No training stats found"}, 404


api.add_resource(UploadParameters, '/upload_parameters')
api.add_resource(StartTraining, '/start_training')
api.add_resource(GetTrainingStats, '/get_training_stats')

if __name__ == '__main__':
    app.run(debug=True)
