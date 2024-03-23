import pytest
import logging
import tracemalloc
from unittest.mock import patch, MagicMock

from datasets import load_dataset

from Training import app, StartTraining, GetTrainingStats

logging.basicConfig(filename='./testTraining.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testDataUpload.log', mode='a')
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


def test_upload_parameters(client):
    datasets = load_dataset("food101", split="train[:5000]")
    datasets.save_to_disk('/home/thai/training_test/')
    rv = client.post('/upload_parameters', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'parameters': {'num_train_epochs': 3, 'per_device_train_batch_size': 16, 'per_device_eval_batch_size': 64,
                       'warmup_steps': 500, 'weight_decay': 0.01},
        'train_dataset': '/home/thai/training_test/'
    })
    assert b'Parameters uploaded successfully' in rv.data



def test_start_training(client):


    rv = client.post('/start_training', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'model_name': 'google/vit-base-patch16-224-in21k'
    })
    assert b'Training for model test_model completed successfully' in rv.data


def test_get_training_stats(client):
    rv = client.get('/get_training_stats/test_user/test_project/test_model')
    assert rv.status_code == 200
    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()

# if __name__ == "__main__":
#   pytest.main()
