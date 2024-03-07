import pytest
from flask import Flask
from Dataupload import app, db, image_collection, ImageUpload, LabelUpload
from io import BytesIO

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_image_upload(client):
    # Mock the file for upload
    data = {
        'file': (BytesIO(b'my file contents'), 'test_file.jpg')
    }

    # Send POST request
    response = client.post('/upload_images/user1/project1', content_type='multipart/form-data', data=data)

    # Assert the response
    assert response.status_code == 200
    assert response.data == b'Image uploaded successfully'

    # Assert the file was saved in the database
    image_data = image_collection.find_one({'user_id': 'user1', 'project_id': 'project1'})
    assert image_data is not None
    assert image_data['filename'] == '/path/to/save/test_file.jpg'
    assert image_data['label'] is None

def test_label_upload(client):
    # Mock the file for upload
    data = {
        'file': (BytesIO(b'my file contents'), 'test_label.jpg')
    }

    # Send POST request
    response = client.post('/upload_label/user1/project1', content_type='multipart/form-data', data=data)

    # Assert the response
    assert response.status_code == 200
    assert response.data == b'Label uploaded successfully'

    # Assert the label was saved in the database
    image_data = image_collection.find_one({'user_id': 'user1', 'project_id': 'project1'})
    assert image_data is not None
    assert image_data['label'] == 'test_label.jpg'
