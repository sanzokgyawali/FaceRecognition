import cv2


class FaceHaar:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

    def detect_frontal_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bodies = self.face_classifier.detectMultiScale(gray, 1.2, 3)

        result = []
        for (x, y, w, h) in bodies:
            result.append([x, y, x + w, y + h])

        return result
