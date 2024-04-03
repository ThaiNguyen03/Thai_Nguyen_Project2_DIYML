# Description

This is the inference module for the project 2. It should generate a unique endpoint depending on user_id and project_id.
This module implements a queue to process inference requests. Currently only models built using PyTorch are supported.

# Usage
- "/inference/<user_id>/<project_id>": The user can send a POST request containing the path to the trained model and the image path, and the API will run inference using the model saved at the model_path on the image saved at image_path.