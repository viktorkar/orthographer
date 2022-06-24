import cv2 as cv
import numpy as np

class ONNXModel():
    def __init__(self, model_path):
        self.recognizer = cv.dnn.readNet(model_path)
        
     ####################################################################################################
    def __decodeText(self, scores):
        text = ""
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        total_conf = 1.0
        for i in range(scores.shape[0]):
            probs = np.exp(scores[i][0])
            conf = np.max(probs) / (np.sum(probs) + 1e-50) # Add a small number to avoid division with 0.
            total_conf *= conf
            c = np.argmax(scores[i][0])
            if c != 0:
                text += alphabet[c - 1]
            else:
                text += '-'

        # adjacent same letters as well as background text must be removed to get the final output
        char_list = []
        for i in range(len(text)):
            if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
                char_list.append(text[i])

        return ''.join(char_list), total_conf

    ####################################################################################################
    def __call__(self, cropped):
        """Finds the text in a cropped out image of a word.
           Returns the inf time, the found word and the prediction confidence.
        """
        # Create a 4D blob from cropped image
        cropped = cv.resize(cropped, (100, 32)) # Model expects this size.
        blob = cv.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
        self.recognizer.setInput(blob)

        # Run the recognition model
        r = self.recognizer.forward()

        # Decode text and add prediction to ocr_dict.
        text, conf = self.__decodeText(r)
        
        return text, conf
