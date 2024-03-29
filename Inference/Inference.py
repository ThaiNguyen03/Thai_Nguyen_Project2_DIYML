from flask import Flask, request
from flask_restful import Resource, Api
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from queue import Queue
from threading import Thread, Event

task_complete_event = Event()
app = Flask(__name__)
api = Api(app)
task_queue = Queue()
results_dict = {}


def inference(data):
    model_path = data.get('model_path')
    image_path = data.get('image_path')
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        return {"message": "Image not found"}, 404

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    inputs = image_processor(pil_image, return_tensors="pt")

    my_model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
    with torch.no_grad():
        logits = my_model(**inputs).logits
    predicted = logits.argmax(-1).item()

    return {"message": "Inference run successfully", "results": my_model.config.id2label[predicted]}, 200


def worker(request_id):
    while not task_queue.empty():
        data = task_queue.get()
        result, status_code = inference(data)
        print(result)
        results_dict[request_id] = (result, status_code)

        task_queue.task_done()
    task_complete_event.set()


class InferenceAPI(Resource):
    def post(self, user_id, project_id):
        data = request.get_json()
        request_id = f"{user_id}_{project_id}"
        task_queue.put(data)
        task_complete_event.clear()
        worker_thread = Thread(target=worker, args=(request_id,))
        worker_thread.start()
        worker_thread.join()
        #task_complete_event.wait()
        if request_id in results_dict:
            result, status_code = results_dict.pop(request_id)
            return result, status_code
        else:
            return {"message": "Error occurred during inference"}, 500


api.add_resource(InferenceAPI, '/inference/<string:user_id>/<string:project_id>')

if __name__ == '__main__':
    app.run(debug=True)
