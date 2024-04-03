# Description

This is the data upload module of project 2.

# Usage

- "/ImageUpload": Users can send a POST requests to upload images and DELETE requests to remove images based on their image_id from the database
- "/LabelUpload": Users can send a POST request to this endpoint to update the label of an image based on their image_id, user_id and project_id
- "ParquetExport": Users can send a GET request to this endpoint to compile all the images and labels uploaded into a Parquet file in the local directory of the API deployment environment.

# Note
Current unit tests are conducted using mock requests from the unittest.mock library. 