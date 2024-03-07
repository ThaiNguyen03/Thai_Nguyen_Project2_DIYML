from flask import Flask, request
from flask_restful import Resource, Api
from pymongo import MongoClient
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
app = Flask(__name__)
api = Api(app)
# Initialize MongoDB client
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
model_collection = db['parameters']  # Collection to store image metadata

class StartTraining(Resource):
    def post(self):
        data = request.get_json()
        model_name = data.get('model')
        project_id = data.get('project_id')
        parameters = model_collection.find_one({"project_id": project_id})
        # load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = ...
        eval_dataset = ...
        # load training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=parameters.get('num_train_epochs'),
            per_device_train_batch_size=parameters.get('per_device_train_batch_size'),
            per_device_eval_batch_size=parameters.get('per_device_eval_batch_size'),
            warmup_steps=parameters.get('warmup_steps'),
            weight_decay=parameters.get('weight_decay'),
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        trainer.train()

        return {"message": model_name}, 200

class GetTrainingStats(Resource):
    def get(self):
        data = request.get_json()
        model_name = data.get('model')
        # Assume you have the trained model available
        # model = YourTrainedModel()

        return model_name, 200

api.add_resource(StartTraining, '/start_training')
api.add_resource(GetTrainingStats, '/get_training_stats')

if __name__ == '__main__':
    app.run(debug=True)
