from flask import Flask, request, jsonify
import os
import pandas as pd

app = Flask(__name__)

class Training:
    def __init__(self, user_id):
        self.id = user_id
    def configure_training(self, parameters):
        self.parameters = parameters

    def start_training(self, model, data, parameters):
        model = model.fit(data, parameters)
        return model

    def get_training_stats(self, model,data):
        # Implement your logic to get training stats
        return model.score(data)

training = Training()

@app.route('/configure_training', methods=['POST'])
def configure_training():
    parameters = request.json
    training.configure_training(parameters)
    return jsonify({'message': 'Training configured successfully'}), 200

@app.route('/get_training_stats', methods=['POST, GET'])
def get_training_stats():
    stats = training.get_training_stats()
    return jsonify(stats), 200

if __name__ == '__main__':
    app.run(debug=True)
