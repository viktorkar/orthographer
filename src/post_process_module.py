import cv2 as cv
from neuspell.corrector_subwordbert import BertChecker
from symspell import SymSpellClass
from util import concat_boxes_to_image, convert_wordlist_to_string
from trocr import TrOCR
from symspellpy import editdistance

class PostProcessModule:
    """The module used for post processing the text recognition results. Can be changed to use 
    different post processing algorithms as long as run() returns the same format."""
    def __init__(self, pp_type, language):
        self.tickmeter = cv.TickMeter() # Used to calculate inference times.
        self.pp_type = pp_type
        self.max_concats = 10           # Maximum number of concatenated words when running TrOCR.
        self.rec_min_conf = 0.20         # Minimum confidence for the text recognition model.
        self.language = language

        self.algorithm = editdistance.DistanceAlgorithm(2)
        self.ed = editdistance.EditDistance(self.algorithm)
        
        self.__setup()
    
    ####################################################################################################
    def __setup(self):
        """Setup the post processing module. Makes sure the processor support the used language."""
        if self.pp_type == 'trocr' and self.language == 'en':      # TrOCR is only available for English.
            self.processor = TrOCR()
        elif self.pp_type == 'neuspell' and self.language == 'en': # NeuSpell is only available for English.
            self.processor = BertChecker()
            self.processor.from_pretrained()
        elif self.pp_type == 'symspell':                           # Symspell is available for all languages.
            self.processor = SymSpellClass(self.language)
        else:
            self.processor = None

    ####################################################################################################
    def __improve_predictions_trocr(self, rec_pred, cropped_images, max_box_height, ocr_dict):
        """Improves the predictions of the text recognition module using TrOCR."""
        concat_images, lengths = concat_boxes_to_image(cropped_images, max_box_height, self.max_concats)
        trocr_pred = self.processor(concat_images, lengths)

        final_pred = []
        for dict_value, (rec_string, rec_conf), trocr_string in zip(ocr_dict.values(), rec_pred, trocr_pred):
            # If recognition system and TrOCR finds the same word or if the
            # confidence of the prediction is really high, we say the prediction is correct.
            #print("Recognized: {}, Confidence: {}, TrOCR: {}\n".format(rec_string, rec_conf, trocr_string))
            if rec_string == trocr_string or rec_conf >= self.rec_min_conf:
                final_pred.append(rec_string)
                continue
            
            # Else we check how similar the two predictions are. If the similarity is high,
            # use the TrOCR prediction.
            longest_string_length = max(len(trocr_string), len(rec_string))
            diff = self.ed.compare(rec_string, trocr_string, longest_string_length)
            if diff < 3:
                final_pred.append(trocr_string)
                dict_value[1].append((trocr_string, self.rec_min_conf))
            else:
                final_pred.append(rec_string)

        return final_pred

    ####################################################################################################
    def __improve_predictions_symspell(self, rec_pred, ocr_dict):
        """Improves the predictions of the text recognition module using Symspell."""
        final_pred = []
        for dict_value, (rec_string, rec_conf) in zip(ocr_dict.values(), rec_pred):
            if rec_conf >= self.rec_min_conf:
                final_pred.append(rec_string)
                continue

            suggestion = self.processor.get_suggestion(rec_string)
            if suggestion:
                final_pred.append(suggestion)
                dict_value[1].append((suggestion, self.rec_min_conf))
            else:
                final_pred.append(rec_string)
            
        return final_pred

    ####################################################################################################
    def __improve_predictions_neuspell(self, rec_pred, ocr_dict):
        """Improves the predictions of the text recognition module using NeuSpell."""
        final_pred = []
        
        if len(rec_pred) == 0:
            return 0, final_pred

        rec_pred_string = convert_wordlist_to_string([x[0] for x in rec_pred])
        neuspell_string = self.processor.correct(rec_pred_string)

        for dict_value, (rec_string, rec_conf), neuspell_suggestion in zip(ocr_dict.values(), rec_pred, neuspell_string.split(' ')):
            # If recognition system and BertChecker finds the same word or if the
            # confidence of the prediction is really high, we say the prediction is correct.
            if rec_string == neuspell_suggestion or rec_conf >= self.rec_min_conf:
                final_pred.append(rec_string)
            else:
                # Else we check how similar the two predictions are. If the similarity is high,
                # use the BertChecker prediction.
                longest_string_length = max(len(neuspell_suggestion), len(rec_string))
                diff = self.ed.compare(rec_string, neuspell_suggestion, longest_string_length)
                if diff < 3:
                    final_pred.append(neuspell_suggestion)
                    dict_value[1].append((neuspell_suggestion, self.rec_min_conf))
                else:
                    final_pred.append(rec_string)

        return final_pred
    
    ####################################################################################################
    def update_language(self, language):
        """Update the language of the post processing module and creates a new instance of it."""
        if self.language != language:
            self.language = language
            self.__setup()

    ####################################################################################################
    def run(self, cropped_images, max_box_height, rec_pred, ocr_dict):
        """Runs the post processing module which tries to improve predictions with low confidence.
        Updates the ocr_dict with the improved predictions.

        Args:
            cropped_images: A list of cropped images.
            max_box_height: The height of the tallest box.
            rec_pred: A list of tuples containing the predictions and confidence of the recognition model.
            ocr_dict: A dictionary containing information about previous predictions.

        Returns:
            pp_infTime: The inference time to run the post processing.
            pred_strings: A list of the processed text strings.
        """
        self.tickmeter.start()

        # Improve predictions if we have set some post processing.
        if self.pp_type == 'trocr':
            pred_strings = self.__improve_predictions_trocr(rec_pred, cropped_images, max_box_height, ocr_dict)
        elif self.pp_type == 'symspell':
            pred_strings = self.__improve_predictions_symspell(rec_pred, ocr_dict)
        elif self.pp_type == 'neuspell':
            pred_strings = self.__improve_predictions_neuspell(rec_pred, ocr_dict)
        else:
            pred_strings = [x[0] for x in rec_pred]

        self.tickmeter.stop()
        pp_infTime = self.tickmeter.getTimeMilli()
        self.tickmeter.reset()
        
        return pp_infTime, pred_strings
