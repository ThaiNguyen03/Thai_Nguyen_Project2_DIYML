import pytest
import logging
import tracemalloc
import PIL
from PIL import Image
from Inference import app, InferenceAPI
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


def test_inference_api(client):
    test_set = load_dataset("food101", split="validation[:100]")
    # test_set.save_to_disk('./training_test')
    test_set.to_parquet('./inference.parquet')
    image = test_set["image"][10]
    image_path = "./inference_data"
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    image = image.save(f"{image_path}/image.png")
    response = client.post('/inference', json={
        'model_name': 'test_model_path',
        'image_path': f'{image_path}/image.png',
        'model_path': '../Training/test_user/test_project/model'
    })
    assert response.status_code == 200
    assert response.get_json() == {"message": "Inference run successfully", "results": "beignets"}
def test_wrong_image(client):
        test_set = load_dataset("food101", split="validation[:100]")
        # test_set.save_to_disk('./training_test')
        test_set.to_parquet('./inference.parquet')
        image = test_set["image"][10]
        image_path = "./inference_data"
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        image = image.save(f"{image_path}/image.png")
        response = client.post('/inference', json={
            'model_name': 'test_model_path',
            'image_path': f'{image_path}/image_wrong.png',
            'model_path': '../Training/test_user/test_project/model'
        })
        assert response.status_code == 404

        mylogger.info('Test for successful inference passed.')
current, peak = tracemalloc.get_traced_memory()
mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")


# Stop tracing memory allocations
tracemalloc.stop()
