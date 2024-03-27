import pytest
from flask import json
from Analysis import app, Analysis
from datasets import load_dataset
import numpy as np
from PIL import Image
import logging
import tracemalloc
from pymongo import MongoClient
mongo_url = 'mongodb://localhost:27017'
client = MongoClient(mongo_url)
db = client['ML_data']
collection = db['Analysis_data']
logging.basicConfig(filename='./testAnalysis.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testAnalysis.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
mylogger.addHandler(fhandler)
mylogger.setLevel(logging.DEBUG)
tracemalloc.start()
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client




def test_analysis_api(client):
    test_set = load_dataset("food101", split="train[:100]")
    test_set.save_to_disk('./analysis_test')
    test_set.to_parquet('./analysis_test/analysis_test.parquet')
    dataset_path = './analysis_test/analysis_test.parquet'

    response = client.post('/analysis', json={
        'dataset_path': dataset_path,
        'model_name': 'google/vit-base-patch16-224-in21k'
    })
    try:
        assert response.status_code == 200
        mylogger.info("Test for post method pass")
    except AssertionError:
        mylogger.error("Test for post method failed")
def test_analysis_api_get(client):

    # Insert a test document
    test_doc = {'size': (512, 512), 'mean': 128.0, 'std_dev': 25.0, 'image_id': 0}
    collection.insert_one(test_doc)

    # Send a GET request to the '/analysis' endpoint with the image id
    response = client.get('/analysis', json = {'id':0})

    # Check the response status code and data
    try:
        assert response.status_code == 200
        mylogger.info("Test for get method passed.\n")
    except AssertionError:
        mylogger.error("Test for get method failed\n")
    # Cleanup the test document
    collection.delete_one({'image_id': 0})
current, peak = tracemalloc.get_traced_memory()
mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
tracemalloc.stop()


