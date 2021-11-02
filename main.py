from face_engine import FaceEngine
from setting import *
from send_request_grid import publish_event
from datetime import datetime
from face_haar import FaceHaar
import blob_control
import _thread as thread
import pytz
import time
import func
import cv2
import sys
import os


class FaceProcess:

    def __init__(self, camera_list):
        self.class_face = FaceEngine()
        self.class_face_haar = FaceHaar()
        self.db_data_list = []
        self.db_name_list = []
        self.db_feature_list = []
        self.db_unidentified_name_list = []
        self.db_unidentified_feature_list = []

        self.load_database()
        self.load_unidentified_db()

        if not os.path.isdir(FOLDER_UNIDENTIFIED):
            os.mkdir(FOLDER_UNIDENTIFIED)

        self.camera_list = camera_list
        self.cap_list = []
        self.video_capture_list = []
        self.frame_list = []
        self.update_frame = []
        self.ret_image = []
        self.frame_ind_list = []
        self.face_result = []
        self.event_data_list = []
        self.azure_store_img_list = []
        self.azure_store_feature_list = []
        self.video_size_list = []

        for i in range(len(camera_list)):
            self.cap_list.append(cv2.VideoCapture(camera_list[i]))
            self.frame_list.append(None)
            self.update_frame.append(False)
            self.ret_image.append(None)
            self.face_result.append({FACE_COORDINATES: [], FACE_SCORES: [], FACE_NAMES: []})
            self.frame_ind_list.append(0)

            width = int(self.cap_list[i].get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_FACTOR)
            height = int(self.cap_list[i].get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_FACTOR)
            self.video_size_list.append([width, height])
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            self.video_capture_list.append(cv2.VideoWriter('result{}.avi'.format(i), fourcc, 30.0, (width, height)))

    def load_database(self):
        # ----------------------- load the database ----------------------------
        db_data_list = []
        db_name_list = []
        db_feature_list = []

        if os.path.isfile(DB_CSV):
            db_data = func.read_text(DB_CSV)
            db_data_list = db_data.splitlines()
            for ii in range(len(db_data_list)):
                cell = db_data_list[ii].split(',')
                db_name_list.append(cell[0])
                db_feature_list.append([])
                for jj in range(2, len(cell)):
                    db_feature_list[-1].append(float(cell[jj]))
        else:
            print("\tCouldn't find the database file.")

        self.db_data_list = db_data_list
        self.db_name_list = db_name_list
        self.db_feature_list = db_feature_list

    def load_unidentified_db(self):
        # ----------------------- load the database ----------------------------
        db_name_list = []
        db_feature_list = []

        if os.path.isfile(DB_UNIDENTIFIED_CSV):
            db_data = func.read_text(DB_UNIDENTIFIED_CSV)
            db_data_list = db_data.splitlines()
            for ii in range(len(db_data_list)):
                cell = db_data_list[ii].split(',')
                db_name_list.append(cell[0])
                db_feature_list.append([])
                for jj in range(1, len(cell)):
                    db_feature_list[-1].append(float(cell[jj]))
        else:
            print("\tCouldn't find the unidentified database file.")

        self.db_unidentified_name_list = db_name_list
        self.db_unidentified_feature_list = db_feature_list

    @staticmethod
    def get_valid_detection(face_aligned_list, face_pos_list, face_score_list):
        face_list = []
        coordinate_list = []
        score_list = []
        for i in range(len(face_score_list)):
            if face_score_list[i] > DETECTION_THRESHOLD:
                face_list.append(face_aligned_list[i])
                coordinate_list.append(face_pos_list[i])
                score_list.append(face_score_list[i])

        return face_list, coordinate_list, score_list

    def add_user(self, img_file, face_name):
        """
            register face image and features into database folder
        """
        if not os.path.isdir(DB_PATH):
            os.mkdir(DB_PATH)

        if not os.path.isdir(DB_IMAGES_PATH):
            os.mkdir(DB_IMAGES_PATH)

        img = cv2.imread(img_file)

        if img is None:
            msg = "Image file isn't exist!"
            return False, msg

        face_list, coordinate_list, score_list = self.class_face.detect_face(img)
        img_aligned, _, _ = self.get_valid_detection(face_list, coordinate_list, score_list)
        feature_list = self.class_face.get_feature(img_aligned)

        if len(feature_list) == 0:
            msg = "Face doesn't detected!"
            return False, msg

        img_db_name = datetime.now().strftime("%m-%d-%Y_%H_%M_%S_%f") + '_' + os.path.split(img_file)[-1]
        cv2.imwrite(os.path.join(DB_IMAGES_PATH, img_db_name), img)

        feature_db_line = face_name + ',' + img_db_name
        for i in range(len(feature_list[0])):
            feature_db_line += ',' + str(round(feature_list[0][i], 8))
        func.append_text(DB_CSV, feature_db_line + '\n')

        msg = "Successfully added!"
        return True, msg

    def del_user(self, user_name):
        """
            remove user info from db
        """
        new_db_data = ''
        ret = False
        msg = "No found this user!"
        for i in range(len(self.db_data_list)):
            cells = self.db_data_list[i].split(',')
            if cells[0] == user_name:
                func.rm_file(os.path.join(DB_IMAGES_PATH, cells[1]))
                ret = True
                msg = "Delete user successfully!"
            else:
                new_db_data += self.db_data_list[i] + '\n'

        func.write_text(DB_CSV, new_db_data)

        return ret, msg

    def check_image(self, img):
        """
            Check the image and recognize the face
        """
        img_ret = img.copy()
        # --------------- detect the face and feature from file ----------------
        face_aligned_list, face_pos_list, face_score_list = self.class_face.detect_face(img_ret)
        face_list, coordinate_list, _ = self.get_valid_detection(face_aligned_list, face_pos_list, face_score_list)

        # ------------------------ recognize the face ---------------------------
        feature_list = self.class_face.get_feature(face_list)

        name_list = []
        score_list = []
        for i in range(len(feature_list)):
            max_val = 0
            max_name = 'None'
            for j in range(len(self.db_feature_list)):
                _, score = self.class_face.compare(feature_list[i], self.db_feature_list[j])
                if score > max_val:
                    max_val = score
                    max_name = self.db_name_list[j]

            if max_val <= RECOGNITION_THRESHOLD:
                max_name = 'Unknown'

            name_list.append(max_name)
            score_list.append(max_val)

        # ------------------------- draw the result image --------------------------
        for i in range(len(coordinate_list)):
            pos = coordinate_list[i]
            cv2.rectangle(img_ret, (pos[0], pos[1]), (pos[2], pos[3]), (255, 0, 0), 2)
            if i < len(name_list) and name_list[i] != "Unknown":
                text_face = '{}, {}'.format(name_list[i], round(score_list[i], 2))
            else:
                text_face = ''

            cv2.putText(img_ret, text_face, (pos[0], pos[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return coordinate_list, name_list, score_list, img_ret, feature_list

    def check_image_file(self, img_file):
        img = cv2.imread(img_file)
        img_h, img_w = img.shape[:2]
        if max(img_h, img_w) > 5000:
            sc = 8
        elif max(img_h, img_w) > 4000:
            sc = 6
        elif max(img_h, img_w) > 3000:
            sc = 4
        elif max(img_h, img_w) > 2000:
            sc = 3
        elif max(img_h, img_w) > 1400:
            sc = 2
        elif max(img_h, img_w) > 1000:
            sc = 1.5
        else:
            sc = 1

        img = cv2.resize(img, None, fx=1/sc, fy=1/sc)
        coordinates, names, scores, img_ret, _ = self.check_image(img)

        for i in range(len(coordinates)):
            print("\tFace detected => Name: {}, Score: {}".format(names[i], round(scores[i], 2)))

        cv2.imwrite('result.jpg', img_ret)
        if SHOW_VIDEO:
            cv2.imshow('result', img_ret)
            cv2.waitKey(0)

    def check_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        cnt = 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_FACTOR)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_FACTOR)
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        vid_capture = cv2.VideoWriter('result.avi', fourcc, 15.0, size)

        while True:
            ret, frame = cap.read()
            cnt += 1
            if not ret:
                break

            if RESIZE_FACTOR != 1.0:
                frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            if cnt % 2 == 0:
                continue

            coordinate_list, name_list, score_list, img, _ = self.check_image(frame)

            if SHOW_VIDEO:
                cv2.imshow('result', img)

            if SAVE_VIDEO:
                vid_capture.write(img)

            if cv2.waitKey(10) == ord('q'):
                break

        vid_capture.release()
        cap.release()

    def read_frame(self, camera_ind, scale_factor=1.0):
        while True:
            ret, frame = self.cap_list[camera_ind].read()
            if ret:
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
                self.frame_list[camera_ind] = frame
                self.update_frame[camera_ind] = True
            else:
                cam_url = self.camera_list[camera_ind]
                print("Fail to read camera!", cam_url)
                self.cap_list[camera_ind].release()
                time.sleep(0.5)
                self.cap_list[camera_ind] = cv2.VideoCapture(cam_url)

            time.sleep(0.02)

    def process_frame(self, camera_ind):
        while True:
            if self.update_frame[camera_ind]:
                if self.frame_list[camera_ind] is not None:
                    self.frame_ind_list[camera_ind] += 1
                    img_color = self.frame_list[camera_ind].copy()

                    # ---------------- detect the faces and analyse -------------------
                    coordinates, names, scores, img_ret, feature_list = self.check_image(img_color)
                    self.face_result[camera_ind][FACE_COORDINATES] = coordinates
                    self.face_result[camera_ind][FACE_SCORES] = scores
                    self.face_result[camera_ind][FACE_NAMES] = names

                    # ------------------- detect the frontal faces ---------------------
                    if len(scores) > 0 and min(scores) < STORE_THRESHOLD:
                        frontal_faces = self.class_face_haar.detect_frontal_face(img_color)
                    else:
                        frontal_faces = []

                    # -------------- select the faces which need to store azure --------
                    for i in range(len(scores)):
                        if scores[i] >= STORE_THRESHOLD:
                            continue

                        new_x1 = max(0, coordinates[i][0] - 15)
                        new_y1 = max(0, coordinates[i][1] - 15)
                        new_x2 = min(self.video_size_list[camera_ind][0], coordinates[i][2] + 15)
                        new_y2 = min(self.video_size_list[camera_ind][1], coordinates[i][3] + 15)
                        img_crop = img_color[new_y1:new_y2, new_x1:new_x2]

                        for j in range(len(frontal_faces)):
                            if func.check_overlap_rect(coordinates[i], frontal_faces[j]):
                                [x1, y1, x2, y2] = frontal_faces[j]
                                cv2.rectangle(img_ret, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                self.azure_store_img_list.append(img_crop)
                                self.azure_store_feature_list.append(feature_list[i])
                                break

                    if SEND_EVENT:
                        if len(self.face_result[camera_ind][FACE_NAMES]) > 0:
                            data = {
                                'timestamp': str(datetime.now(pytz.timezone('US/Central'))),
                                'camera_id': func.get_camera_id(self.camera_list[camera_ind]),
                                'detection': str(self.face_result[camera_ind])
                            }
                            self.event_data_list.append(data)

                    self.ret_image[camera_ind] = img_ret

                # initialize the variable
                self.update_frame[camera_ind] = False

            time.sleep(0.01)

    def send_event_data(self):
        while True:
            if len(self.event_data_list) == 0:
                time.sleep(0.1)
            else:
                data = self.event_data_list.copy()
                self.event_data_list = []
                publish_event(data)

    def store_unidentified_face(self):
        while True:
            if len(self.azure_store_img_list) == 0:
                time.sleep(0.1)
            else:
                # compare with the registered unidentified data

                max_val = 0
                max_name = 'None'
                f_store = True
                for j in range(len(self.db_unidentified_name_list)):
                    _, score = self.class_face.compare(self.azure_store_feature_list[0], self.db_unidentified_feature_list[j])
                    if score > max_val:
                        max_val = score
                        max_name = self.db_unidentified_name_list[j]

                if max_val <= RECOGNITION_THRESHOLD:
                    unidentified_index = str(len(self.db_unidentified_name_list) + 1)
                    img_name = unidentified_index + '_' + str(time.time()) + '.jpg'
                    self.db_unidentified_name_list.append(unidentified_index)
                    self.db_unidentified_feature_list.append(self.azure_store_feature_list[0])

                    feature_db_line = unidentified_index
                    for i in range(len(self.azure_store_feature_list[0])):
                        feature_db_line += ',' + str(round(self.azure_store_feature_list[0][i], 8))
                    func.append_text(DB_UNIDENTIFIED_CSV, feature_db_line + '\n')
                else:
                    blob_list = blob_control.get_container_list()
                    cnt = 0
                    for i in range(len(blob_list)):
                        if blob_list[i].startswith(max_name + '_'):
                            cnt += 1

                    if cnt > 5:
                        f_store = False

                    img_name = max_name + '_' + str(time.time()) + '.jpg'

                if f_store:
                    # save to local folder
                    img_path = os.path.join(FOLDER_UNIDENTIFIED, img_name)
                    cv2.imwrite(img_path, self.azure_store_img_list[0])

                    # store images to azure
                    blob_control.upload_blob(img_path, img_name)

                self.azure_store_img_list.pop(0)
                self.azure_store_feature_list.pop(0)
                print("Unidentified face is stored to Azure storage!")

    def check_video_thread(self):
        if SEND_EVENT:
            thread.start_new_thread(self.send_event_data, ())

        if SEND_FACES:
            thread.start_new_thread(self.store_unidentified_face, ())

        for i in range(len(self.cap_list)):
            thread.start_new_thread(self.read_frame, (i, RESIZE_FACTOR))
            thread.start_new_thread(self.process_frame, (i, ))

        while True:
            for cam_ind in range(len(self.cap_list)):
                if self.frame_list[cam_ind] is not None:
                    if DISPLAY_DETECT_FRAME_ONLY:
                        if self.ret_image[cam_ind] is not None:
                            if SHOW_VIDEO:
                                cv2.imshow('org' + str(cam_ind), self.ret_image[cam_ind])

                            if SAVE_VIDEO:
                                self.video_capture_list[cam_ind].write(self.ret_image[cam_ind])
                    else:
                        if self.frame_list[cam_ind] is not None:
                            img_org = self.frame_list[cam_ind].copy()
                            result = self.face_result[cam_ind].copy()
                            for i in range(len(result[FACE_COORDINATES])):
                                pos = result[FACE_COORDINATES][i]
                                name_list = result[FACE_NAMES]
                                score_list = result[FACE_SCORES]
                                cv2.rectangle(img_org, (pos[0], pos[1]), (pos[2], pos[3]), (255, 0, 0), 2)
                                if i < len(name_list) and name_list[i] != "Unknown":
                                    text_face = '{}, {}'.format(name_list[i], round(score_list[i], 2))
                                else:
                                    text_face = ''

                                cv2.putText(img_org, text_face, (pos[0], pos[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 255, 0), 2)

                            if SHOW_VIDEO:
                                cv2.imshow('org' + str(cam_ind), img_org)

                            if SAVE_VIDEO:
                                self.video_capture_list[cam_ind].write(img_org)

            if cv2.waitKey(10) == ord('q'):
                break

        for cam_ind in range(len(self.cap_list)):
            self.cap_list[cam_ind].release()
            self.video_capture_list[cam_ind].release()


if __name__ == '__main__':
    in_args = ['check',
               # '../samples/1.mp4',
               '',
               # '../images/24.jpg',
               # 'webcam'
               '',
               ]

    for arg_ind in range(1, len(sys.argv)):
        in_args[arg_ind - 1] = sys.argv[arg_ind]

    class_main = FaceProcess(CAMERA_URL)

    if in_args[0].lower() == 'add':
        print("Add user to database")
        ret_add, msg_add = class_main.add_user(in_args[1], in_args[2])
        print('\t', msg_add)
    elif in_args[0].lower() in ['del', 'remove']:
        print("Delete user from database")
        ret_del, msg_del = class_main.del_user(in_args[1])
        print('\t', msg_del)
    elif in_args[0].lower() == 'check':
        if in_args[1][-4:].lower() in ['.jpg', 'jpeg', '.png', '.bmp']:
            print("Checking image ...")
            class_main.check_image_file(in_args[1])
        elif in_args[1][-4:].lower() in ['.mp4', '.avi', 'mpeg']:
            print("Checking video ...")
            class_main.check_video(in_args[1])
        elif in_args[1] == 'webcam':
            print("Checking webcam ...")
            class_main.check_video(0)
        elif in_args[1] == '':
            class_main.check_video_thread()
