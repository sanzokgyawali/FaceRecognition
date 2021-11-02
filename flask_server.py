from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from setting import *
from datetime import datetime
from main import FaceProcess
import func
import _thread as thread
import os


class FaceApi(Resource):

    def __init__(self):
        self.store_data = {}

    def post(self):
        request_json = request.get_json()

        if request_json['action'] == 'add':
            print("Received user register request!")
            req_img_data = request_json['image']
            req_user = request_json['user']
            data_filename = 'temp_{}.jpg'.format(datetime.now().microsecond)
            func.img_decode_b64(req_img_data, data_filename)

            ret, msg = class_face.add_user(data_filename, req_user)
            class_face.load_database()
            func.rm_file(data_filename)

            result_json = {'state': ret,
                           'message': msg}

        elif request_json['action'] == 'del':
            print("Received user delete request!")
            req_user = request_json['user']
            ret, msg = class_face.del_user(req_user)
            class_face.load_database()

            result_json = {'state': ret,
                           'message': msg}

        else:
            print("Received unknown request!")
            result_json = {'state': False,
                           'message': 'Unrecognized request!'}

        return jsonify(dict(result=result_json))


app = Flask(__name__)
api = Api(app)
api.add_resource(FaceApi, '/face_service/v1.0')


# ------- run main face engine for camera urls ---------
class_face = FaceProcess(CAMERA_URL)
if SEND_EVENT:
    thread.start_new_thread(class_face.send_event_data, ())

if SEND_FACES:
    thread.start_new_thread(class_face.store_unidentified_face, ())

for i in range(len(CAMERA_URL)):
    thread.start_new_thread(class_face.read_frame, (i, RESIZE_FACTOR))
    thread.start_new_thread(class_face.process_frame, (i,))


if __name__ == '__main__':

    app.run(host="0.0.0.0",
            port=int(os.environ.get("PORT", 3000)),
            debug=False,
            threaded=False)
