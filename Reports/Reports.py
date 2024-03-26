from flask import Flask, request
from flask_restful import Resource, Api
from pymongo import MongoClient
import numpy as np
app = Flask(__name__)
api = Api(app)
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
model_collection = db['model_data']  # Collection to store model parameters
stats_collection = db['stats']  # Collection to store training stats


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
api.add_resource(GetTrainingStats, '/get_training_stats')

if __name__ == '__main__':
    app.run(debug=True)