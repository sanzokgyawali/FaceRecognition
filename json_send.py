
import requests
import base64
import json
import sys


def make_request_json_add(img_file, name):

    file_data = open(img_file, 'rb')

    json_data = {
        "action": 'add',
        "image": base64.b64encode(file_data.read()).decode('UTF-8'),
        "user": name
    }

    return json_data


def make_request_json_del(name):
    json_data = {
        "action": 'del',
        "user": name
    }

    return json_data


def send_request(server, req_json):

    response = requests.post(url=server, json=req_json)
    print("Server responded with %s" % response.status_code)

    response_json = response.json()
    return response_json


if __name__ == '__main__':

    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = '../samples/Adam.jpg'

    # url_server = 'http://localhost:3000/face_service/v1.0'
    url_server = 'http://52.142.14.71:3000/face_service/v1.0'

    # json_request = make_request_json_add(img_file=filename, name="Adam")
    json_request = make_request_json_del(name="Adam")

    ret_response = send_request(url_server, json_request)

    print(json.dumps(ret_response, indent=4))
