import pytest
import logging
import tracemalloc
from unittest.mock import patch, MagicMock
from flask import Flask
from Dataupload import app, ImageUpload, LabelUpload  # replace with the actual name of your Flask app

logging.basicConfig(filename='./testDataUpload.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testDataUpload.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
mylogger.addHandler(fhandler)
mylogger.setLevel(logging.DEBUG)
tracemalloc.start()


def test_image_upload():
    with app.test_request_context():
        with patch('flask.request') as mock_request:
            mock_file = MagicMock()
            mock_file.filename = 'test_image.jpg'
            mock_request.files = {'file': mock_file}

            try:
                response = app.test_client().post('/upload_images/test_user/test_project/test_image')
                assert response.status_code == 200
                assert b'Image uploaded successfully' in response.data
                mylogger.info('Image upload test passed successfully.')
            except Exception as e:
                mylogger.error(str(e))

    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()


def test_label_upload():
    with app.test_request_context():
        with patch('flask.request') as mock_request:
            mock_file = MagicMock()
            mock_file.filename = 'test_label.txt'
            mock_request.files = {'file': mock_file}

            try:
                response = app.test_client().post('/upload_label/test_user/test_project/test_image')
                assert response.status_code == 200
                assert b'Label uploaded successfully' in response.data
                mylogger.info('Label upload test passed successfully.')
            except Exception as e:
                mylogger.error(str(e))

    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()


if __name__ == "__main__":
    pytest.main()
