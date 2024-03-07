import pytest
from flask import json
from Training import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_start_training(client):
    # Test the StartTraining endpoint
    response = client.post('/start_training', json={'model': 'test_model', 'project_id': 'test_project_id'})
    assert response.status_code == 200
    assert response.get_json() == {"message": "test_model"}

def test_get_training_stats(client):
    # Test the GetTrainingStats endpoint
    response = client.get('/get_training_stats', json={'model': 'test_model'})
    assert response.status_code == 200
    assert response.get_json() == "test_model"
