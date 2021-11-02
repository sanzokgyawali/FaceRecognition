import face_recognition_net
import tensorflow as tf
import cv2
import numpy as np
import os
from scipy import spatial
from tensorflow.python.platform import gfile


class FaceEngine:

    def __init__(self):
        model_path = 'model'
        # face detection model
        with tf.Graph().as_default():
            sess_face = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            with sess_face.as_default():
                self.pnet, self.rnet, self.onet = face_recognition_net.create_mtcnn(sess_face, model_path)

        # face feature model
        with tf.Graph().as_default():
            self.sess_feature = tf.Session()
            model_exp = os.path.expanduser(os.path.join(model_path, '20180408-102900.pb'))
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, input_map=None, name='')

            # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    @staticmethod
    def __prewhiten__(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    def detect_face(self, img_org, image_size=160, margin=11):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        img_list = []
        coordinate_list = []
        score_list = []
        img = img_org[:, :, ::-1]
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = face_recognition_net.detect_face(img,
                                                             minsize=minsize,
                                                             pnet=self.pnet,
                                                             rnet=self.rnet,
                                                             onet=self.onet,
                                                             threshold=threshold,
                                                             factor=factor)

        for i in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[i, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = cv2.resize(cropped[:, :, ::-1], (image_size, image_size))[:, :, ::-1]
            prewhitened = self.__prewhiten__(aligned)
            img_list.append(prewhitened)
            coordinate_list.append([int(bounding_boxes[i][0]),
                                    int(bounding_boxes[i][1]),
                                    int(bounding_boxes[i][2]),
                                    int(bounding_boxes[i][3])])
            score_list.append(bounding_boxes[i][4])

        return img_list, coordinate_list, score_list

    @staticmethod
    def compare(emb1, emb2, threshold=0.7):
        score = 1 - spatial.distance.cosine(emb1, emb2)
        return score > threshold, score

    def get_feature(self, images):
        # Run forward pass to calculate embeddings
        if len(images) == 0:
            return []
        else:
            feed_dict = {self.images_placeholder: images,
                         self.phase_train_placeholder: False}
            emb = self.sess_feature.run(self.embeddings, feed_dict=feed_dict)

            return emb


if __name__ == '__main__':
    class_face = FaceEngine()

    file_images = ['../samples/both.jpg', '../samples/both2.jpg', '../samples/Stan.jpg']
    img1 = cv2.imread(file_images[0])
    img2 = cv2.imread(file_images[1])
    img_aligned1, detect_boxes1, scores1 = class_face.detect_face(img1)
    img_aligned2, detect_boxes2, scores2 = class_face.detect_face(img2)

    print(detect_boxes1)
    print(scores1)

    embeddings1 = class_face.get_feature(img_aligned1)
    embeddings2 = class_face.get_feature(img_aligned2)

    print(class_face.compare(embeddings1[0], embeddings2[1]))
    print(class_face.compare(embeddings1[1], embeddings2[0]))
