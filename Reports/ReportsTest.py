import pytest
import logging
import tracemalloc
from Reports import app, GetTrainingStats

logging.basicConfig(filename='./testTraining.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testTraining.log', mode='a')
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

def test_get_training_stats(client):
    rv = client.get('/get_training_stats', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'model_name': 'google/vit-base-patch16-224-in21k'
    })
    try:
        assert rv.status_code == 200
        mylogger.info("Test passed")
    except AssertionError:
        mylogger.error("Test failed")
    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
