import cv2 as cv2
import numpy as np
import time
import robomaster


kpx = 0.02
kpy = 0.02
kpz = 20.0
lower = np.array([0, 200, 40], dtype=np.uint8)
upper = np.array([10, 255, 255], dtype=np.uint8)

from robomaster import robot
if __name__ == '__main__':
    tl_drone = robot.Drone()
    tl_drone.initialize()

    tl_flight = tl_drone.flight

    print("Initiating movement")
    tl_camera = tl_drone.camera
    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("low")
    tl_camera.set_resolution("high")
    tl_camera.set_bitrate(6)
    tl_flight.takeoff().wait_for_completed()
    print('Done')
    while True:
        img = tl_camera.read_cv2_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        img2 = cv2.bitwise_and(gray, gray, mask=mask)
        gray = 255-img2
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 2)
        thresh = 255 - thresh

        kernel = np.ones((5, 5), np.uint8)
        rect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)

        # thin
        kernel = np.ones((5, 5), np.uint8)
        rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)

        contours = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = None
        for c in contours:
            area_thresh = 0
            area = cv2.contourArea(c)
            if area > area_thresh:
                area = area_thresh
                big_contour = c

        rot_bbox = img.copy()
        error = np.array([0, 0, 0])
        z_error = 0
        if big_contour is not None:
            rot_rect = cv2.minAreaRect(big_contour)
            box = cv2.boxPoints(rot_rect)
            box = np.int0(box)
            area = cv2.contourArea(big_contour)
            cv2.drawContours(rot_bbox, [box], 0, (0, 0, 255), 2)
            error = np.array([[480, 360], [480, 360], [480, 360], [480, 360]]) - box
            error = error.sum(axis=0)
            z_error = np.log(60000/area)
        print(error[0] * kpx, error[1] * kpy, z_error)
        tl_flight.rc(a=-error[0]*kpx, b=z_error*kpz, c= error[1]*kpy, d=0)
        cv2.imshow('Thresh', rot_bbox)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    tl_flight.rc(a=0, b=0, c=0, d=0)
    cv2.destroyAllWindows()
    tl_camera.stop_video_stream()
    tl_flight.land().wait_for_completed()
    tl_drone.close()
