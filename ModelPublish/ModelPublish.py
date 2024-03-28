from flask import Flask, request
from flask_restful import Resource, Api
from pymongo import MongoClient
import os
import shutil

app = Flask(__name__)
api = Api(app)

mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
stats_collection = db['stats']  # Collection to store training stats
publish_model = db['publish_model']  # Collection to store published models


class PublishModel(Resource):
    def post(self):
        data = request.get_json()
        model_id = data.get('model_id')
        user_id = data.get('user_id')
        project_id = data.get('project_id')

        # Search for the model in stats_collection
        model_entry = stats_collection.find_one({"model_id": model_id})
        if not model_entry:
            return {"message": f"Model with ID {model_id} not found"}, 404

        # Create a new folder for the published model if it doesn't exist
        publish_folder = f"./published_models/{user_id}/{project_id}"
        if os.path.exists(publish_folder):
            shutil.rmtree(publish_folder)

        # Copy the content from model_saved_path to the publish folder
        model_saved_path = model_entry['model_saved_path']
        shutil.copytree(model_saved_path, publish_folder)


        try:
            publish_model.insert_one({
                'user_id': user_id,
                'project_id': project_id,
                'model_id': model_id,
                'published_folder_path': os.path.abspath(publish_folder)
            })
        except Exception as e:
            return {"message": "Model data not published"}, 404

        return {"message": f"Model {model_id} published successfully"}, 200


api.add_resource(PublishModel, '/publish_model')

if __name__ == '__main__':
    app.run(debug=True)
