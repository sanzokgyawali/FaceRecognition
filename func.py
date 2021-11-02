
import os
import base64
import shutil
import csv
import numpy as np


def rm_file(file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)


def rm_tree(folder_name):
    shutil.rmtree(folder_name, True)
    os.mkdir(folder_name)


def img_encode_b64(img_name):
    img_file = open(img_name, 'rb')
    img_b64_data = base64.b64encode(img_file.read()).decode('UTF-8')
    return img_b64_data


def img_decode_b64(img_date, img_name):
    img_out = open(img_name, 'wb')
    img_out.write(base64.b64decode(img_date))
    img_out.close()


def get_distance(vec1, vec2):
    d = 0
    for i in range(len(vec1)):
        d += (float(vec1[i]) - float(vec2[i])) ** 2

    if d < 0.4:
        dist = d / 6
    else:
        dist = d * (6.0 / 7) + 1.0 / 7

    return dist


def get_match_value(vec1, vec2):
    distance = get_distance(vec1, vec2)
    return 100 * (max(1 - distance, 0.0))


def write_text(filename, text):
    file1 = open(filename, 'w')
    file1.write(text)
    file1.close()


def append_text(filename, text):
    file1 = open(filename, 'a')
    file1.write(text)
    file1.close()


def read_text(filename):
    file1 = open(filename, 'r')
    text = file1.read()
    file1.close()

    return text


def get_file_list(root_dir):
    path_list = []
    file_list = []
    join_list = []
    for path, _, files in os.walk(root_dir):
        for name in files:
            path_list.append(path)
            file_list.append(name)
            join_list.append(os.path.join(path, name))

    return path_list, file_list, join_list


def load_csv(filename):
    """
        load the csv data and return it.
    """
    if not os.path.isfile(filename):
        return []

    file_csv = open(filename, 'r')
    reader = csv.reader(file_csv)
    data_csv = []
    for row_data in reader:
        data_csv.append(row_data)

    file_csv.close()
    return data_csv


def save_csv(filename, data):
    """
        save the "data" to filename as csv format.
    """
    file_out = open(filename, 'wb')
    writer = csv.writer(file_out)
    writer.writerows(data)
    file_out.close()


def append_csv(filename, data):
    file_out = open(filename, 'a')
    writer = csv.writer(file_out)
    writer.writerows(data)
    file_out.close()


def get_emotion_features(lm):
    land_rot = np.rot90(lm)

    norm_x = normalizing_vector(land_rot[0], 100)
    norm_y = normalizing_vector(land_rot[1], 100)

    return norm_x + norm_y


def normalizing_vector(vec, d_max):
    min_vec = min(vec)
    max_vec = max(vec)
    a = float(d_max) / (max_vec-min_vec)
    b = float(d_max) * min_vec / (min_vec-max_vec)
    new_vec = []
    for i in range(len(vec)):
        new_val = a * vec[i] + b
        new_vec.append(int(new_val))

    return new_vec


def check_overlap_rect(rect1, rect2):
    min_x1 = min(rect1[0], rect1[2])
    max_x1 = max(rect1[0], rect1[2])
    min_y1 = min(rect1[1], rect1[3])
    max_y1 = max(rect1[1], rect1[3])

    min_x2 = min(rect2[0], rect2[2])
    max_x2 = max(rect2[0], rect2[2])
    min_y2 = min(rect2[1], rect2[3])
    max_y2 = max(rect2[1], rect2[3])

    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False
    else:
        return True


def get_camera_id(cam_url):
    """
        get channel index 5 from the url "rtsp://admin:pass@1.2.3.4:554/cam/realmonitor?channel=5&subtype=0"
    """
    ind1 = cam_url.find('channel=')
    if ind1 == -1:
        return '-1'

    new_string = cam_url[ind1 + 8:]
    ind2 = new_string.find('&')
    if ind2 == -1:
        return '-1'

    str_id = new_string[:ind2]

    if str_id.isdigit():
        return str_id
    else:
        return '-1'
