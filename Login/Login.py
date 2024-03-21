from flask import Flask, request
from flask_restful import Resource, Api
from pymongo import MongoClient


app = Flask(__name__)
api = Api(app)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['user_data']
collection = db['users']


class LoginAPI(Resource):
    def post(self):
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        # Check if user exists
        user = collection.find_one({'username': username})
        if user:
            # Check the password
            if user['password']== password:
                return {"message": "Login successful"}, 200
            else:
                return {"message": "Wrong password"}, 401
        else:
            return {"message": "User not found"}, 404


api.add_resource(LoginAPI, '/login')

if __name__ == '__main__':
    app.run(debug=True)
