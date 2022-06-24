from onnx_model import ONNXModel
from dtrb.model_builder import create_dtrb_model
import cv2 as cv

from util import find_highest_prob_pred

class TextRecognitionModule:
    """The module used for text recognition. The module can be changed to use different text recognizers as long
    as run() is implemented in the same way."""
    def __init__(self, rec_type, use_dict=True):
        self.tickmeter = cv.TickMeter()
        self.rec_type = rec_type
        self.max_nbr_pred = 5   # Maximum number of predictions for each word.
        self.use_dict = use_dict
        self.__setup()
        
    ####################################################################################################
    def __setup(self):
        """Setup the text recognition module."""
        if self.rec_type == "resnet":
            self.recognizer = ONNXModel("../models/TR/ResNet_CTC.onnx")
        elif self.rec_type == "dtrb":
            self.recognizer = create_dtrb_model()
        
    ####################################################################################################
    def run(self, cropped_images, ocr_dict=[]):
        """Perform text recognition on a given list of images. Uses ocr_dict to connect predictions
        to previous and future predictions.

        Args:
            cropped_images: A list of cropped images.
            ocr_dict: A dictionary containing information about previous predictions.
            use_dict: Set to false to run model without previous predictions.
        Returns:
            rec_infTime: The inference time to run the text recognition.
            result: A list of recognized text strings.
        """
        self.tickmeter.start()
        result = []

        if self.use_dict:
            for idx, value in enumerate(ocr_dict.values()):
                cropped = cropped_images[idx]
                counts = value[0]
                preds = value[1]

                # If we have predicted this string self.max_nbr_pred times, add the word with highest confidence
                # to result and delete all other predictions from the list to speed up next iteration.
                if counts >= self.max_nbr_pred:
                    best_pred, best_conf = find_highest_prob_pred(preds)
                    value[1] = [(best_pred, best_conf)]
                    result.append((best_pred, best_conf))
                    continue
                
                pred, conf = self.recognizer(cropped)

                value[1].append((pred, conf))
                result.append(find_highest_prob_pred(value[1]))
        else:
            """
            This case is used for benchmark-notebook/text-recognition-benchmark.ipynb
            To ensure that recognition_module has the same behaviour as onnx_model and trocr,
            cropped_images should be only one image, when calling run with self.use_dict set to false.
            """
            result, conf = self.recognizer(cropped_images)
            
        self.tickmeter.stop()
        rec_infTime = self.tickmeter.getTimeMilli()
        self.tickmeter.reset()

        return rec_infTime, result



