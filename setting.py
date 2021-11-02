import os


# -------------------- camera --------------------
# CAMERA_URL = ['rtsp://username:password@IP_ADDR:554/cam/realmonitor?channel=1&subtype=0']
# CAMERA_URL = ['../sample.mp4']
# CAMERA_URL = ['../car1.mp4', '../1.mp4']
CAMERA_URL = [
    'rtsp://admin:alphaventus98470@103.1.94.175:554/cam/realmonitor?channel=3&subtype=0',
    'rtsp://admin:alphaventus98470@103.1.94.175:1154/cam/realmonitor?channel=5&subtype=0',
    'rtsp://admin:alphaventus98470@103.1.94.175:1054/cam/realmonitor?channel=10&subtype=0',
    'rtsp://admin:alphaventus98470@103.1.94.175:954/cam/realmonitor?channel=9&subtype=0',
    'rtsp://admin:alphaventus98470@103.1.94.175:854/cam/realmonitor?channel=1&subtype=0',
    'rtsp://admin:alphaventus98470@103.1.94.175:654/cam/realmonitor?channel=2&subtype=0'
]

# --------------------- DB ----------------------
DB_PATH = 'database'
DB_CSV = os.path.join(DB_PATH, 'register.csv')
DB_IMAGES_PATH = os.path.join(DB_PATH, 'images')

FOLDER_UNIDENTIFIED = 'save_unidentified'
DB_UNIDENTIFIED_CSV = os.path.join(FOLDER_UNIDENTIFIED, 'unregistered.csv')

# --------------- Face detection ----------------
DETECTION_THRESHOLD = 0.99
RECOGNITION_THRESHOLD = 0.6
STORE_THRESHOLD = 0.25

# -------------------- Others -------------------
RESIZE_FACTOR = 1.0
DISPLAY_DETECT_FRAME_ONLY = True
FACE_COORDINATES = 'coordinates'
FACE_SCORES = 'scores'
FACE_NAMES = 'names'
SAVE_VIDEO = False
SHOW_VIDEO = False

# ----------------- Azure Action ------------------
SEND_EVENT = True
SEND_FACES = True
