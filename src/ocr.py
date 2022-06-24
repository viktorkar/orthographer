import cv2 as cv

from detection_module   import TextDetectionModule
from recognition_module import TextRecognitionModule
from post_process_module import PostProcessModule
from spellcheck_module  import SpellCheckModule

from util import crop_images
####################################################################################################
class OCR:
    def __init__(self, args, img_width, img_height, language):
        self.args = args        # Arguments from the command line.
        self.ocr_dict = {}      # Passes information between iterations and modules.

        self.detector      = TextDetectionModule(self.args.td_model, img_width, img_height)
        self.recognizer    = TextRecognitionModule(self.args.tr_model)
        self.postprocessor = PostProcessModule(self.args.post_process, language)
        self.spellchecker  = SpellCheckModule(self.args.spell_check, language)

    ##################################################################################
    def update_language(self, language):
        """Changes how the postprocessor and spellchecker work as they are dependent on the language."""
        self.postprocessor.update_language(language)
        self.spellchecker.update_language(language)

    ##################################################################################
    def perform_ocr(self, frame, queue, text_rec = False):
        """Runs the entire OCR on a given frame. The results are pushed to the given queue."""
        self.text_rec = text_rec
        
        # Convert to gray scale.
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Perform text detection.
        det_infTime, boxes, self.ocr_dict = self.detector.run(frame, self.ocr_dict, text_rec)
        cropped_images, max_box_height = crop_images(frame, boxes)
        
        # Perform text recognition if enabled and we have some cropped words.
        if self.text_rec and cropped_images:
            rec_infTime, rec_pred = self.recognizer.run(cropped_images, self.ocr_dict)

            # Perform post processing.
            pp_infTime, pred_strings = self.postprocessor.run(cropped_images, max_box_height, rec_pred, self.ocr_dict)
            # Perform spellcheck.
            sc_infTime, spellchecks = self.spellchecker.run(pred_strings)
    
            queue.put((det_infTime, rec_infTime, pp_infTime, boxes, pred_strings, spellchecks))

        else:
            queue.put((det_infTime, 0, 0, boxes, [], []))

    ####################################################################################################
    def benchmark_ocr(self, frame, text_rec = True):
        """Runs the entire system given an frame for benchmarking. OCR dict is not used and 
        no queue is required like in the perform_ocr method."""
        self.ocr_dict = {} # Reset dictionary before every frame.
        
        # Convert to gray scale.
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Perform text detection.
        det_infTime, boxes, self.ocr_dict = self.detector.run(frame, self.ocr_dict, text_rec)
        cropped_images, max_box_height = crop_images(frame, boxes)
        
        # Perform text recognition if we have some cropped words.
        if cropped_images:
            rec_infTime, rec_pred = self.recognizer.run(cropped_images, self.ocr_dict)
            pred_strings = [x[0] for x in rec_pred]

            # Perform post processing.
            pp_infTime, final_strings = self.postprocessor.run(cropped_images, max_box_height, rec_pred, self.ocr_dict)
            
            # Perform spellcheck.
            sc_infTime, spellchecks = self.spellchecker.run(final_strings)
    
            return det_infTime, rec_infTime, pp_infTime, sc_infTime, boxes, pred_strings, final_strings, spellchecks
        else:
            return det_infTime, 0, 0, 0, boxes, [], [], []
        
    ####################################################################################################
    def set_minConf(self, min_confidence):
        self.detector.set_minConf(min_confidence)