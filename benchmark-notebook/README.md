# Benchmark Notebooks

## Handwritten text detection benchmark
To evaluate the detection models, the jupyter notebook [text-detection-benchmark.ipynb](text-detection-benchmark.ipynb) runs the models on the IAM Forms (Aachen test split) dataset. Only the handwritten text is extracted from the images, only bounding boxes of words and numbers are used and badly segmented lines are removed. The intersection over union (IOU) is calculated to determine if a prediction is correct or not. If IOU > 0.5, it is considered correct. As metrics, recall and precision are calculated.

## Handwritten text recognition benchmark
To evaluate the recognition models, the jupyter notebook [text-recognition-benchmark.ipynb](text-recognition-benchmark.ipynb) runs the models on the IAM Words (Aachen test split) dataset. Images of punctuation symbols and badly segmented words are removed. As metrics, CER, WER and inference time are calculated.

## Full system benchmark
The file [benchmark.py](../src/benchmark.py) runs the entire system on a custom created [system_dataset](../data/system_dataset) containing spelling errors. Running the file creates a .csv that can be explored in the jupyter notebook [full-system-benchmark.ipynb](fullsystem-benchmark.ipynb). As metrics, CER, inference times and spellcheck F1, recall and precision are calculated.
