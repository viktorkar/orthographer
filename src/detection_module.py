import math
from east import EAST
from yolo import YOLO
import cv2 as cv

class TextDetectionModule:
    """The module used for detecting handwritten text in images. Can be changed to use 
    different types of detection models as long as run() returns the same format."""
    def __init__(self, det_type, img_width, img_height):
        self.tickmeter = cv.TickMeter()
        self.det_type = det_type
        self.conf = 0.5
        self.iou = 0.4
        self.pad_factor = 0.15               # Factor for adaptive padding for bounding boxes.

        self.xscaler = (1 / img_width)  * 40 # Think 40 possible columns for word positions.
        self.yscaler = (1 / img_height) * 60 # Think 60 possible rows for word positions.

        self.__setup()

    ####################################################################################################
    def __setup(self):
        """Setup the detection module."""
        if self.det_type == "yolo":
            self.detector = YOLO(self.conf, self.iou)
        elif self.det_type == "east":
            self.detector = EAST(self.conf, self.iou)

    ####################################################################################################
    def __process_boxes(self, boxes, ocr_dict, text_rec):
        """Processes the detected boxes and connect them to previously made predictions."""
        new_dict = {}
        for box in boxes:
            # Calculate ocr_dict key.
            x_center = box[0] + (box[1] - box[0]) * 0.5     # xmin + width / 2
            y_center = box[2] + (box[3] - box[2]) * 0.5     # ymin + height / 2
            key = (math.floor(x_center * self.xscaler), math.floor(y_center * self.yscaler))
            
            # Add to new dictionary. 
            new_dict[key] = [1, []] # [counter, [recognized strings]]

            # If we have found this box before, and are running text recognition,
            # increment its counter and add prev recognized strings. 
            if key in ocr_dict and text_rec:
                new_dict[key][0] += ocr_dict[key][0]
                new_dict[key][1].extend(ocr_dict[key][1])

        return new_dict

    ####################################################################################################
    def __aston_sort(self, boxes):
        """Sorts the found bounding boxes top->down, left->right. Returns a list of the sorted boxes."""
        boxes_sorted = []
        boxes['ycenter'] = (boxes['ymax'] + boxes['ymin']) / 2
        mean_height = (boxes['ymax'] - boxes['ymin']).mean()
        while not boxes.empty:
            # Find smallest y_center value
            idx = boxes['ycenter'].idxmin()
            ycenter_smallest = boxes.loc[idx]['ycenter']

            # Find boxes on the same line.
            boxes_on_line = boxes[boxes['ycenter'] - ycenter_smallest <= mean_height]

            # Sort by x_min value.
            boxes_on_line = boxes_on_line.sort_values(by='xmin')
            boxes_on_line = boxes_on_line.drop(columns='ycenter')

            boxes_on_line_list = boxes_on_line.values.tolist()
            boxes_sorted.extend(boxes_on_line_list)

            # Delete from boxes
            boxes = boxes.drop(boxes_on_line.index)

        return boxes_sorted

    ####################################################################################################
    def __add_padding(self, boxes, frame_width, frame_height):
        """Adds some padding to each detected bounding box."""
        for box in boxes:
            x_pad = int((box[1] - box[0]) * self.pad_factor)
            y_pad = int((box[3] - box[2]) * self.pad_factor)

            # Calculate new coordinates
            xmin = box[0] - x_pad
            xmax = box[1] + x_pad
            ymin = box[2] - y_pad
            ymax = box[3] + y_pad

            # Set new coordinates but make sure it is not outside of frame.
            box[0] = xmin if xmin >= 0 else box[0] 
            box[1] = xmax if xmax <= frame_width else box[1]
            box[2] = ymin if ymin >= 0 else box[2]
            box[3] = ymax if ymax <= frame_height else box[3]

        return boxes

    ####################################################################################################
    def set_minConf(self, min_confidence):
        self.conf = min_confidence
        self.detector.set_minConf(min_confidence)

    ####################################################################################################
    def set_minIOU(self, iou):
        self.iou = iou
        self.detector.set_nmsThreshold(iou)

    ####################################################################################################
    def run(self, frame, ocr_dict, text_rec):
        """Performs text detection on the given frame.

        Args:
            frame: The image to detect text on.
            ocr_dict: A dictionary containing previous detections.
            text_rec: Flag if text recognition is enabled.

        Returns:
            det_infTime: Inference time for the detection module.
            bounding_boxes: The bounding boxes of the detected words in the frame.
            ocr_dict: An updated ocr_dict with the new detections.
        """
        self.tickmeter.start()
        (H, W) = frame.shape[:2]

        # Detect boxes, sort and add padding.
        bounding_boxes = self.detector(frame)
        bounding_boxes = self.__aston_sort(bounding_boxes)
        bounding_boxes = self.__add_padding(bounding_boxes, frame_width=W, frame_height=H)

        # Connect boxes to ocr_dict and previous predictions.
        ocr_dict = self.__process_boxes(bounding_boxes, ocr_dict, text_rec)

        self.tickmeter.stop()
        det_infTime = self.tickmeter.getTimeMilli()
        self.tickmeter.reset()
    
        return det_infTime, bounding_boxes, ocr_dict