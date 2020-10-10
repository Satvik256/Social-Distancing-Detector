import numpy as np
import cv2

def detect_people(frame, net, ln, personIdx=0):
	(H, W) = frame.shape[:2]
	results = []
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:

		for detection in output:
			scores = detection[5:]
			ID = np.argmax(scores)
			confidence = scores[ID]

			if ID == personIdx and confidence > 0.3 :
				box = detection[0:4] * np.array([W, H, W, H])
				(ctrX, ctrY, Wd, Ht) = box.astype("int")

				x = int(ctrX - (Wd / 2))
				y = int(ctrY - (Ht / 2))

				boxes.append([x, y, int(Wd), int(Ht)])
				centroids.append((ctrX, ctrY))
				confidences.append(float(confidence))

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

	if len(idxs) > 0:

		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	return results