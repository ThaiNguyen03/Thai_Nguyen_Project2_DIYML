# Description

This is the training module for the project 2. It uses the transformers library to create a trainer object to train the model. The API implements a queue to process the user requests.

# Usage
- "/upload_parameters": users can upload their own parameters to pass into the trainer object by sending a POST request to this endpoint
- "/start_training": users can send a POST request with their user_id, project_id, model_name and the path to a training_dataset. The API will then use a model from HuggingFace hub that matches the name provided by the user and trained it with the dataset at the provided path.