import cv2 as cv
import math
import numpy as np

import pandas as pd

####################################################################

class EAST():
    def __init__(self, conf, iou, model_path=None):
        self.width          = 640 # should be a multiple of 32 for the EAST model to work well
        self.height         = 640 # should be a multiple of 32 for the EAST model to work well
        self.min_confidence = conf # Confidence threshold (0-1)
        self.nms_threshold  = iou  # NMS IoU threshold (0-1)   
        
        if model_path:
            self.detector = cv.dnn.readNet(model_path)
        else:
            self.detector = cv.dnn.readNet("../models/TD/frozen_east_text_detection.pb")

        self.layer_names = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

    ####################################################################################################
    def __decodeBoundingBoxes(self, scores, geometry):
        boxes = []
        confidences = []

        height = scores.shape[2]
        width = scores.shape[3]
        
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if (score < self.min_confidence):
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                boxes.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return boxes and confidences
        return [boxes, confidences]

    ####################################################################################################
    def __call__(self, frame):
        (H, W) = frame.shape[:2]
        rW = W / float(self.width)
        rH = H / float(self.height)

        # construct a blob from the image
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        blob = cv.dnn.blobFromImage(frame, 1.0, (self.width, self.height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.detector.setInput(blob)

        # forward pass it to EAST model
        (prob_scores, geometry) = self.detector.forward(self.layer_names)

         # Get scores and geometry
        [boxes, confidences] = self.__decodeBoundingBoxes(prob_scores, geometry)

        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, self.min_confidence, self.nms_threshold)

        # initialize the list of result (boxes)
        result = []
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv.boxPoints(boxes[i])

            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            
            xmin, ymin = np.min(vertices, axis=0)
            xmax, ymax = np.max(vertices, axis=0)

            result.append([xmin, xmax, ymin, ymax])

        # TextDetectionModule expect a panadas DataFrame to be returned.
        return pd.DataFrame(result, columns = ['xmin', 'xmax', 'ymin', 'ymax']).astype('int32')

    ####################################################################################################
    def set_minConf(self, min_confidence):
        self.min_confidence = min_confidence

    ####################################################################################################
    def set_nmsThreshold(self, nms_threshold):
        self.nms_threshold = nms_threshold