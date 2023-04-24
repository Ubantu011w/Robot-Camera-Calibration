# Import Libraries 
import cv2
import numpy as np
from paho.mqtt import client as mqtt_client

def connect_mqtt():
    global client
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client("Abdul")
    client.username_pw_set('ubantu011', 'N/A')
    client.on_connect = on_connect
    client.connect('io.adafruit.com', 1883)

connect_mqtt()

# On clicked
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        string = str(x) + "," + str(y)
        result = client.publish(topic="ubantu011/feeds/calibration", payload=string)
        imaging_model(x, y)
        # print(string)

# forward imaging model opencv
def imaging_model(x, y):
    pixel_point = np.array([[x, y]], dtype=np.float32)
    # Convert the 2D image point to normalized image coordinates using cv2.undistortPoints() 
    norm_point = cv2.undistortPoints(pixel_point, mtx, dist) # Convert the pixel point to normalized image coordinates
    # Convert the normalized image coordinates to a 3D point
    homogeneous_point = np.insert(norm_point, 2, 1)
    # homogeneous_point = np.hstack((norm_point, np.ones((1, 1))))
    rot_matrix, _ = cv2.Rodrigues(rvecs[0].T)
    inv_rot_matrix = np.linalg.inv(rot_matrix)
    inv_trans_matrix = -inv_rot_matrix.dot(tvecs)
    world_point_homogeneous = inv_rot_matrix.dot(homogeneous_point.T) + inv_trans_matrix
    world_point = world_point_homogeneous[:3].T
    print(world_point)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Input "0" means your connected usb webcap. Can be replaced with a RTSP url
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Image Acquisition")

cv2.setMouseCallback("Image Acquisition", click)
imgNum = 0 # For file naming

chessboardSize = (11,11)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32) # prepare object points, like (0,0,0), (1,0,0), etc 
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
while True:
    global mtx, dist, rvecs, tvecs
    ret, frame = cap.read() # ret: boolean
    if not ret:
        print("Cannot Read Frame!!!")
        break
    cv2.imshow("Image Acquisition", frame)
    cv2.setWindowProperty('Image Acquisition', cv2.WND_PROP_TOPMOST, 1) # always on top
    comm = cv2.waitKey(100)
    if comm%256 == 27: # ESC key is pressed
        print("    Command: Ending program..")
        break
    elif comm%256 == 32: # SPACE key is pressed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None) # watch out for chessboardSize
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints = []
            imgpoints = []
            imgNum += 1
            #fileName = "CalibFrame_{}.png".format(imgNum)
            #cv2.imwrite(fileName, frame)
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, chessboardSize, corners2, ret)
            cv2.imshow('Image Acquisition', frame)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            np.save("data/rotationVector" + imgNum, rvecs)
            np.save("data/translationVector" + imgNum, tvecs)
            # print("Camera Calibrated: ", ret)
            print("Camera Matrix: \n", mtx)
            # print("Distortion Parameters: ", dist)
            # print("Rotation Vectors: ", rvecs)
            # print("Translation Vectors: ", tvecs)
            h, w = frame.shape[:2] # get height, width of the frame
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            # undistort
            dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imshow('Image Acquisition', dst)
            cv2.waitKey(2500)
cap.release()
cv2.destroyAllWindows()
