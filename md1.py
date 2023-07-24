# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.restoration import inpaint
import warnings

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

warnings.filterwarnings("ignore")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default='videos/01.avi', help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
vs = cv2.VideoCapture(args["video"])

# while True:
#     frame = vs.read()
#     frame = frame if args.get("video", None) is None else frame[1]
#     cv2.imshow('test', frame)
#     cv2.waitKey(1000)
# exit()

def create_fake_bg(vs):
    firstFrame = None
    bounding_rect = 'undefined'
    for i in range(2):
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        frame = imutils.resize(frame, width=500)
        frameCustomized = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frameCustomized[:, :, 2] += 150
        frameCustomized[:, :, 1] += 50
        frameCustomized[:, :, 0] += 50
        frameCustomized = cv2.cvtColor(frameCustomized, cv2.COLOR_HSV2BGR)
        # cv2.imshow('frame', frameCustomized)
        gray = cv2.cvtColor(frameCustomized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            gray = cv2.GaussianBlur(gray, (221, 221), 0)
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        frameDelta[frameDelta < 45] = 0
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, np.ones((1, 1), dtype=np.uint8), iterations=5)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((4, 4), dtype=np.uint8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((4, 4), dtype=np.uint8))

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # find contours on thresholded image
        bounding_rect = cv2.boundingRect(max(cnts, key=cv2.contourArea))

        if bounding_rect != 'undefined':
            (x, y, w, h) = bounding_rect
            mask = np.zeros(frame.shape[:-1])
            mask[y - 5:y + h + 5, x - 5:x + w + 5] = 1

            frameDefected = frame.copy()
            for layer in range(frameDefected.shape[-1]):
                frameDefected[np.where(mask)] = 0
            #
            # cv2.imshow('test', frameDefected)
            # cv2.waitKey(1000000)
            fakeFrame = inpaint.inpaint_biharmonic(frameDefected, mask, multichannel=True)
            fakeFrame = img_as_ubyte(fakeFrame)
            cv2.imwrite('fake.png', fakeFrame);
            gray = cv2.cvtColor(fakeFrame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            return gray


# initialize the first frame in the video stream
firstFrame = create_fake_bg(vs)

# loop over the frames of the video

while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    bounding_rect = 'undefined'

    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # if the first frame is None, exit
    if firstFrame is None:
        exit

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, np.ones((1, 1), dtype=np.uint8), iterations=2)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # find contours on thresholded image
    # loop over the contours
    largest_area = 0
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue

        area = cv2.contourArea(c)
        if (area > largest_area):
            largest_area = area
            bounding_rect = cv2.boundingRect(c)

    # compute the bounding box for the contour, draw it on the frame,
    # and update the text
    if bounding_rect != 'undefined':
        (x, y, w, h) = bounding_rect
        cv2.rectangle(frame, (x - 5, y - 5), (x + w, y + h), (0, 255, 0), 1)
        text = "Person Detected"

    # show the frame and record if the user presses a key
    cv2.imshow("Original Video", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)

    key = cv2.waitKey(100)

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
