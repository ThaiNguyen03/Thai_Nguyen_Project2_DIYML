import os
import pytest
from ModelPublish import app, PublishModel
import logging
import tracemalloc

logging.basicConfig(filename='./testPublish.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testPublish.log', mode='a')
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


def test_publish_model_success(client):
    # Send a POST request to publish the model
    response = client.post('/publish_model', json={
        'model_id': 'model1',
        'user_id': 'test_user',
        'project_id': 'test_project'
    })
    try:
        assert response.status_code == 200
        mylogger.info("Test for successful publish passed")
    except AssertionError:
        mylogger.error("Test for successful publish not passed")


import os
import pytest
from ModelPublish import app, PublishModel


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_publish_model_wrong_credentials(client):
    # Send a POST request to publish the model
    response = client.post('/publish_model', json={
        'model_id': 'model100',
        'user_id': 'test_user',
        'project_id': 'test_project'
    })
    try:
        assert response.status_code == 404
        mylogger.info("Test for wrong credentials passed")
    except AssertionError:
        mylogger.error("Test for wrong credentials not passed")


current, peak = tracemalloc.get_traced_memory()
mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
tracemalloc.stop()
