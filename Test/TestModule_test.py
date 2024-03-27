import pytest
import logging
import tracemalloc
import PIL
from PIL import Image
from Test import app, TestAPI
import datasets
from datasets import load_dataset
import os

logging.basicConfig(filename='./testInference.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testInference.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
mylogger.addHandler(fhandler)
mylogger.setLevel(logging.DEBUG)

# Start tracing memory allocations
tracemalloc.start()


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_module_api(client):
    test_set = load_dataset("food101", split="validation[:100]")
    test_set.to_parquet('./test.parquet')
    response = client.post('/test', json={
        'model_name': 'test_model_path',
        'dataset_path': './test.parquet',
        'model_path': '../Training/test_user/test_project/model'
    })
    assert response.status_code == 200
    assert 'results' in response.get_json()

    mylogger.info('Test for successful test_module passed.')
import pytest
import logging
import tracemalloc
import PIL
from PIL import Image
from Test import app, TestAPI
import datasets
from datasets import load_dataset
import os

logging.basicConfig(filename='./testInference.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testInference.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
mylogger.addHandler(fhandler)
mylogger.setLevel(logging.DEBUG)

# Start tracing memory allocations
tracemalloc.start()


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_testmodule_wrong_dataset(client):
    test_set = load_dataset("food101", split="validation[:100]")
    test_set.to_parquet('./test.parquet')
    response = client.post('/test', json={
        'model_name': 'test_model_path',
        'dataset_path': './test1.parquet',
        'model_path': '../Training/test_user/test_project/model'
    })
    assert response.status_code == 404


    mylogger.info('Test for check dataset passed.')


# Stop tracing memory allocations
tracemalloc.stop()

current, peak = tracemalloc.get_traced_memory()
mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")


# Stop tracing memory allocations
tracemalloc.stop()
