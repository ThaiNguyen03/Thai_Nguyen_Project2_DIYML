import pytest
import logging
import tracemalloc
from unittest.mock import patch, MagicMock

from datasets import load_dataset

from Training import app, StartTraining, GetTrainingStats

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


def test_upload_parameters(client):
    test_set = load_dataset("food101", split="train[:100]")
    #test_set.save_to_disk('./training_test')
    test_set.to_parquet('./training_test/training_test.parquet')
    rv = client.post('/upload_parameters', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'parameters': {
                        'learning_rate': 5e-5,
                        'per_device_train_batch_size':16,
                        'gradient_accumulation_steps':4,
                        'per_device_eval_batch_size':16,
                        'num_train_epochs': 3,
                        'warmup_ratio':0.1,
                        'logging_steps': 10,
    },
        'train_dataset': './training_test/training_test.parquet'
    })
    assert b'Parameters uploaded successfully' in rv.data



def test_start_training(client):
    rv = client.post('/start_training', json={
        'user_id': 'test_user',
        'project_id': 'test_project',
        'model_name': 'google/vit-base-patch16-224-in21k'
    })
    assert rv.status_code ==200
   # assert b'Training for model test_model completed successfully' in rv.data


def test_get_training_stats(client):
    rv = client.get('/get_training_stats',json = {
        'user_id': 'test_user',
        'project_id': 'test_project',
        'model_name': 'google/vit-base-patch16-224-in21k'
    })
    assert rv.status_code == 200
    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()

# if __name__ == "__main__":
#   pytest.main()
