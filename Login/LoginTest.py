import pytest
import logging
import tracemalloc

from Login import app, LoginAPI, collection  # replace 'your_flask_app' with the name of your python file

# Set up logging
logging.basicConfig(filename='./testLogin.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mylogger = logging.getLogger()
fhandler = logging.FileHandler(filename='testLogin.log', mode='a')
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
        collection.insert_one({
            'username': 'thai',
            'password': '123'
        })
        yield client

def test_login_success(client):
    response = client.post('/login', json={'username': 'thai', 'password': '123'})
    assert response.status_code == 200
    assert response.get_json() == {"message": "Login successful"}
    mylogger.info('Test for successful login passed.')
    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")

def test_login_wrong_password(client):
    response = client.post('/login', json={'username': 'thai', 'password': 'wrong_password'})
    assert response.status_code == 401
    assert response.get_json() == {"message": "Wrong password"}
    mylogger.info('Test for login with wrong password passed.')
    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")

def test_login_user_not_found(client):
    response = client.post('/login', json={'username': 'non_existent_user', 'password': '123'})
    assert response.status_code == 404
    assert response.get_json() == {"message": "User not found"}
    mylogger.info('Test for login with non-existent user passed.')
    current, peak = tracemalloc.get_traced_memory()
    mylogger.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")

# Stop tracing memory allocations
tracemalloc.stop()

# Remove the test user from the database
collection.delete_one({'username': 'thai'})
