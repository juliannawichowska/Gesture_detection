import numpy as np
import cv2
import math

# Creates a color bound based on a ROI
# It analyses the blue square and calculates the maximum, minimum and average HSV values inside the square.
#  Those maximum and minimum values will be used to determine the maximum sensibility possible, and the average will be the color bound used to detect the hand.

def captureCamera(left=False):

    cap = cv2.VideoCapture(0)

    outerRectangleXIni = 300
    outerRectangleYIni = 50
    outerRectangleXFin = 550
    outerRectangleYFin = 300
    innerRectangleXIni = 400
    innerRectangleYIni = 150
    innerRectangleXFin = 450
    innerRectangleYFin = 200

    if left:
        move_to_left = 250
        outerRectangleXIni = outerRectangleXIni - move_to_left
        outerRectangleXFin = outerRectangleXFin - move_to_left
        innerRectangleXIni = innerRectangleXIni - move_to_left
        innerRectangleXFin = innerRectangleXFin - move_to_left

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (outerRectangleXIni, outerRectangleYIni),
                      (outerRectangleXFin, outerRectangleYFin), (0, 0, 0), 0)
        cv2.rectangle(frame, (innerRectangleXIni, innerRectangleYIni),
                      (innerRectangleXFin, innerRectangleYFin), (0, 0, 255), 0)
        cv2.putText(frame, 'Place your hand in the square', (0, 35),
                    font, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            cap.release()
            return None
        elif key != -1:
            roi = frame[innerRectangleYIni +
                        1:innerRectangleYFin, innerRectangleXIni +
                        1:innerRectangleXFin]
            display_result(roi)
            approved = wait_approval()
            if approved:
                break
            cv2.destroyAllWindows()

    cap.release()
    hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array(
        [hsvRoi[:, :, 0].min(), hsvRoi[:, :, 1].min(), hsvRoi[:, :, 2].min()])
    upper = np.array(
        [hsvRoi[:, :, 0].max(), hsvRoi[:, :, 1].max(), hsvRoi[:, :, 2].max()])
    h = hsvRoi[:, :, 0]
    s = hsvRoi[:, :, 1]
    v = hsvRoi[:, :, 2]
    hAverage = np.average(h)
    sAverage = np.average(s)
    vAverage = np.average(v)

    hMax = max(abs(lower[0] - hAverage), abs(upper[0] - hAverage))
    sMax = max(abs(lower[1] - sAverage), abs(upper[1] - sAverage))
    vMax = max(abs(lower[2] - vAverage), abs(upper[2] - vAverage))

    cv2.destroyAllWindows()
    return np.array([[hAverage, sAverage, vAverage],
                     [hMax, sMax, vMax]])


def display_result(roi):
    hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_result = np.concatenate((roi, hsvRoi))
    cv2.imshow('ROI Result', roi_result)


def wait_approval():
    approval = False
    key = cv2.waitKey(0)
    if key != -1 and key != ord('n'):
        approval = True
    return approval


def nothing(x):
    pass

# HSV parameters setting
def apply_sensibility(avg_color, newHSens, newSSens, newVSens, maxSensibility):

    hSens = (newHSens * maxSensibility[0]) / 100
    SSens = (newSSens * maxSensibility[1]) / 100
    VSens = (newVSens * maxSensibility[2]) / 100
    lower_bound_color = np.array(
        [avg_color[0] - hSens, avg_color[1] - SSens, avg_color[2] - VSens])
    upper_bound_color = np.array(
        [avg_color[0] + hSens, avg_color[1] + SSens, avg_color[2] + VSens])
    return np.array([lower_bound_color, upper_bound_color])

 # possibiliy to change HSV values to better adapt to light [%]
def start(avg_color, max_sensibility, video=True, path=None, left=False):

    h = 100
    s = 100
    v = 100

    apply_sensibility(avg_color, h, s, v,
                      max_sensibility)

    cv2.namedWindow('Gesture recognition')
    cv2.createTrackbar('H', 'Gesture recognition', h, 100, nothing)
    cv2.createTrackbar('S', 'Gesture recognition', s, 100, nothing)
    cv2.createTrackbar('V', 'Gesture recognition', v, 100, nothing)

    if path != None:
        frame = cv2.imread(path)
        hand_detection(frame, lower_bound_color, upper_bound_color, left)
        cv2.waitKey(0)
    else:
        video_capture = cv2.VideoCapture(0)

        while True:
            try:
                _, frame = video_capture.read()
                frame = cv2.flip(frame, 1)

                # get values from bars
                newHSens = cv2.getTrackbarPos('H', 'Gesture recognition')
                newSSens = cv2.getTrackbarPos('S', 'Gesture recognition')
                newVSens = cv2.getTrackbarPos('V', 'Gesture recognition')

                # apply the new sensibility values
                lower_bound_color, upper_bound_color = apply_sensibility(
                    avg_color, newHSens, newSSens, newVSens, max_sensibility)

                hand_detection(frame, lower_bound_color, upper_bound_color,
                               left)

            except Exception as e:
                print (e)
                pass

            if not video:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break

            key = cv2.waitKey(10)
            if key == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                break

      # setting the ROI on the left side of the screen
def hand_detection(frame, lower_bound_color, upper_bound_color, left):
    kernel = np.ones((3, 3), np.uint8)

    if left:
        roi = frame[100:300, 100:300]
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 0), 0)
    else:
        roi = frame[50:300, 300:550]
        cv2.rectangle(frame, (300, 50), (550, 300), (0, 0, 0), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    binary_mask = cv2.inRange(hsv, lower_bound_color, upper_bound_color)
    mask = cv2.dilate(binary_mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.GaussianBlur(mask, (5, 5), 90)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
    try:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        l = analyse_defects(cnt, roi)
        analyse_contours(frame, cnt, l + 1)
    except ValueError:
        pass

    show_results(binary_mask, mask, frame)

   # Calculates how many convexity defects are on the image. Defects represent the division between fingers.
def analyse_defects(cnt, roi):
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    l = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            d = (2 * ar) / a

            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)
            cv2.line(roi, start, end, [0, 0, 255], 2)
    return l

  # Write to the image the signal of the hand.
def analyse_contours(frame, cnt, l):
    hull = cv2.convexHull(cnt)

    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)

    arearatio = ((areahull - areacnt) / areacnt) * 100

    font = cv2.FONT_HERSHEY_PLAIN
    if l == 1:
        if areacnt < 2000:
            cv2.putText(frame, 'Place hand in the box', (0, 50), font, 2,
                        (255, 255, 255), 3, cv2.LINE_AA)
        else:
            if arearatio < 12:
                cv2.putText(frame, 'Start presentation', (0, 50), font, 2, (0, 0, 0), 3,
                            cv2.LINE_AA)
            elif arearatio < 17.5:
                cv2.putText(frame, 'Start presentation', (0, 50), font, 2, (0, 0, 0), 3,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Prev slide', (0, 50), font, 2, (0, 0, 0), 3,
                            cv2.LINE_AA)
    elif l == 2:
        cv2.putText(frame, 'Next slide', (0, 50), font, 2, (0, 0, 0), 3, cv2.LINE_AA)
    elif l == 3 or l == 5:

        cv2.putText(frame, 'Stop presentation', (0, 50), font, 2, (0, 0, 0), 3,
                        cv2.LINE_AA)

def show_results(binary_mask, mask, frame):
    combine_masks = np.concatenate((binary_mask, mask), axis=0)
    height, _, _ = frame.shape
    _, width = combine_masks.shape
    masks_result = cv2.resize(combine_masks, dsize=(width, height))
    masks_result = cv2.cvtColor(masks_result, cv2.COLOR_GRAY2BGR)
    result_image = np.concatenate((frame, masks_result), axis=1)
    cv2.imshow('Gesture recognition', result_image)


def main():
    captureCamera()
    lower_color = np.array([0, 50, 120], dtype=np.uint8)
    upper_color = np.array([180, 150, 250], dtype=np.uint8)

    start(lower_color, upper_color)


if __name__ == '__main__':
    main()

