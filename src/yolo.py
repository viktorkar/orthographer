import torch

class YOLO:
    def __init__(self, conf=0.5, iou=0.4, model_path=None):
        if model_path:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path="../models/TD/yolov5s_orthographer.pt")

        self.model.conf = conf          # Confidence threshold (0-1)
        self.model.iou = iou            # NMS IoU threshold (0-1)

    ####################################################################################################
    def __call__(self, frame):
        """Detect text in a given frame and return the bounding boxes of each word."""
        # Pass frame through model, convert to panda dataframe and filter out boxes below confidence threshold.
        result = self.model(frame)
        result = result.pandas().xyxy[0]
        result = result[result['confidence'] > self.model.conf]

        return result.get(['xmin', 'xmax', 'ymin', 'ymax']).astype('int32')

    ####################################################################################################
    def set_nmsThreshold(self, nms_threshold):
        self.model.iou = nms_threshold

    ####################################################################################################
    def set_minConf(self, min_confidence):
        self.model.conf = min_confidence