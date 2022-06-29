# Available Models
The finetuned models can be downloaded from the [google drive folder here](https://drive.google.com/drive/folders/1jRYy1_Cs7pLex1zoNjcQNYo363Kt2Mt6?usp=sharing).

## Text Detection Models
The YOLOv5 models were finetuned on parts of the IAM Forms dataset. The data preprocessing steps are available in the jupyter notebook [IAM_yolo_preprocess.ipynb](data-preprocessing/IAM_yolo_preprocess.ipynb). As can be seen, lines marked with *bad segmentation* were removed from the data, images were reshaped to 640x640 and only bounding boxes of words and numbers were used (not punctuation symbols). The models were trained on the Aachen train+val splits. 

The EAST model was downloaded from https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV.
It has not been finetuned for handwritten text detection.

## Text Recognition Models
The finetuned recognition models were based on the repositiory https://github.com/clovaai/deep-text-recognition-benchmark. The data preprocessing steps are available in the jupyter notebook [lmdb_processing.ipynb](data-preprocessing/lmdb_processing.ipynb)
The models were finetuned using the IAM Words and CVL Words datasets. Because the models can't recognize punctuation symbols, all images containing only puctuation symbols were removed. From the IAM Words dataset, words marked with *bad segmentation* were not used and the models were trained on the Aachen train split. All images were reshaped or padded to match the input size 32x100.

DenseNet_CTC, ResNet_CTC and CRNN_VGG_BiLSTM_CTC were downloaded from https://docs.opencv.org/4.x/d9/d1e/tutorial_dnn_OCR.html and were not finetuned for handwritten text recognition.
