from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2 as cv
import re
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrOCR():
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten',  pad_token_id=self.processor.tokenizer.pad_token_id)
        self.model.to(device)

    ####################################################################################################
    def __call__(self, concat_images, lengths):        
        """Runs text recognition on each image in concat_images. Lengths is the number of words in each image."""
        trocr_pred = []
        for image, length in zip(concat_images, lengths):
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB) # frame to RGB

            # Run recognition.
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            generated_ids = self.model.generate(pixel_values, min_length=length, max_length=length+2)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Filter out everything that isn't words, numbers or spaces. Split on white spaces and add to trocr_pred.
            text = re.sub(r'[^\w\s]', '', text.lower())
            text = text.split(" ")

            # Make sure the number of words is the same as length.
            # We add a blank space if a word is missing or remove words if there are too many.
            nbr_words = len(text)
            if (nbr_words > length):
                text = text[0:length]
            elif (len(text) < length):
                text = text + [""] * (length - nbr_words)

            trocr_pred.extend(text)

        return trocr_pred
