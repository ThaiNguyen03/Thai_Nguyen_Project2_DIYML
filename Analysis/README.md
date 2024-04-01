# Description
This is the Analysis module of the project, which allows users to get preliminary analysis of the images before training.
The module uses the AutoImageProcessor from the transformers library to provide the mean, standard deviation and size of each image.
The API normalizes and extracts the data from each image in the process.

## Inputs
User can send a POST request with the name of the model they are planning on using for training and the path to the dataset on the local directory. 
The API will store the image parameters into a database.

Users can send a GET request with the image_id to get the analysis data of a particular image.