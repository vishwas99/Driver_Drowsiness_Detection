from scipy.spatial import distance as dist #to compute euclidean distance b/w eye points
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread #we need another thread to print sound
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib #to detect and localize face landmarks
import cv2


def sound_alarm(path):
	playsound.playsound(path)


# we need to compute the ratio between the distances b/w vertical eye landmarks to the distances b/w the horizontal eye landmarks
def eye_aspect_ratio(eye):

	# distance b/w 2 pairs of vertical eye landmarks
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	#distance b/w horizontal eye landmarks
	C = dist.euclidean(eye[0], eye[3])

	e_a_r = (A + B)/(2.0 * C)

	return e_a_r

#The above value changes only in small ranges when eyes are open but it changes rapidly when closing


#argparser to parse arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="", help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

#to get command line arguments into the script file


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRES = 0.3
EYE_AR_CONSEC_FRAMES = 48

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off

COUNTER = 0
ALARM_ON = False

#initialize the dlib's face detector (HOG-based) and create the facial landmark predictor

print("Loading Facial landmarks")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#grab indexes of facial landmarks of left eye and right eye

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#now that we got the indices of both eyes we will slice these indices from the entire list produced by dlib

print("Start video streaming thread")

vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

#loop over frames from video stream
while True:

	#grab the frame from video stream and resize and convert it into grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#detect faces
	rects = detector(gray, 0)

	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		#averaging both eye aspect ratios together to obtain a better estimation 

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter

		if ear < EYE_AR_THRES:
			COUNTER += 1

			#if eyes are closed for sufficient time then sound alarm

			if COUNTER > EYE_AR_CONSEC_FRAMES:
				#if alarm not on turn it on

				if not ALARM_ON:

					ALARM_ON = True

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background

					if args['alarm']!="":

						t = Thread(target = sound_alarm,

								args = (args["alarm"],)

							)
						t.deamon = True # A thread can be flagged as a "daemon thread". The significance of this flag is that the entire Python program exits when only daemon threads are left


						t.start()

				#draw alarm on frame
				cv2.putText(frame, "DROWSINESS ALERT!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:

			COUNTER = 0
			ALARM_ON = False

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
























