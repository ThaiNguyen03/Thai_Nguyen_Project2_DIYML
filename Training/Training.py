from flask import Flask, request
from flask_restful import Resource, Api
from pymongo import MongoClient

import torch
from datasets import load_from_disk, load_dataset
import json, os
from transformers import Trainer, DefaultDataCollator, TrainingArguments, AutoModelForImageClassification, AutoTokenizer, \
    AutoFeatureExtractor

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
        # load model and tokenizer

        # tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels)

        # load datasets
        # train_dataset = load_from_disk(f'{user_id}/{project_id}/images/image_train')
        # eval_dataset = load_from_disk(f'{user_id}/{project_id}/images/image_eval')
        api_dataset = load_from_disk(train_dataset_path)
        api_dataset = api_dataset.train_test_split(test_size=0.2)
        labels = api_dataset["train"].features["labels"].names
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        model = AutoModelForImageClassification.from_pretrained(model_name,
                                                                num_labels=len(labels),
                                                                id2label=id2label,
                                                                label2id=label2id
                                                                )
        # eval_dataset = load_dataset('parquet', data_files='/home/thai/training_test/validation-00000-of-00003.parquet')
        # load training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/{user_id}/{project_id}',
            num_train_epochs=parameters.get('num_train_epochs'),
            evaluation_strategy="steps",
            per_device_train_batch_size=parameters.get('per_device_train_batch_size'),
            warmup_steps=parameters.get('warmup_steps'),
            weight_decay=parameters.get('weight_decay'),
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
    def get(self, user_id, project_id, model_name):
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
