
# Orthographer: Real-Time Detection of Spelling Mistakes in Handwritten Notes. <!-- omit in toc --> 

Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [Additional optional dependencies](#additional-optional-dependencies)
- [Models](#models)
- [Data](#data)
  - [IAM](#iam)
  - [CVL](#cvl)
  - [IMGUR5K](#imgur5k)
  - [System dataset](#system-dataset)
- [Running the application](#running-the-application)
  - [Arguments](#arguments)
- [Benchmarking of models](#benchmarking-of-models)
  - [Text detection benchmark](#text-detection-benchmark)
    - [Acquired results](#acquired-results)
  - [Text recognition benchmark](#text-recognition-benchmark)
    - [Acquired results](#acquired-results-1)
  - [Full system and spell correction](#full-system-and-spell-correction)
    - [Arguments](#arguments-1)
    - [Acquired results](#acquired-results-2)
- [Finetuning YOLOv5 for handwritten text detection](#finetuning-yolov5-for-handwritten-text-detection)
  - [Preparing the data](#preparing-the-data)
  - [Finetuning the model](#finetuning-the-model)
- [Acknowledgements](#acknowledgements)
  - [Reference](#reference)
  - [Contact](#contact)
  - [License](#license)

# Introduction 
Orthographer is an end-to-end application for real-time detection of spelling mistakes in handwritten notes. The system was created as a part of [this](https://lup.lub.lu.se/student-papers/search/publication/9110650) master thesis project in 2022.

To perform the task, the system runs a pipeline consisting of modules for video capturing, text detection, text recognition, post processing and finally spell correction. The pipeline is displayed in the image below.

 ![System Architecture](/misc/sys_arch.png)

# Installation
We recommend using [Anaconda](https://www.anaconda.com/distribution/) or the light-weight version [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to make sure that the correct dependencies are installed. Once installed, create an environment by running the following command. The environment will contain all libraries needed to run the program with default settings. By default, the environment will be named `orthographer`, but you can choose another name using the `-n` option.

**OBS**. If running only on CPU, edit environment.yml to avoid installing GPU dependencies.

    $ conda env create -f environment.yml

Next, activate the environment:

    $ conda activate orthographer

## Jupyter Notebooks
To run the conda environment in jupyter notebook, register the environment by running:

    $ python -m ipykernel install --user --name=orthographer

Now the environment can be selected in "Kernel > Change kernel" after starting jupyter notebook.

# Models
The system uses trained machine learning models for handwritten text detection and handwritten text recognition. All finetuned models can be found in the [google drive folder here](https://drive.google.com/drive/folders/1p09AueBHQz16Miw9YkIWDLHjyx_uH3Kx?usp=sharing). Acquired performances of the models are given under [Benchmarking of models](#benchmarking-of-models) and more information can be found in [models/README.md](models/README.md).

Recommended file structure.
```
sc-hwn-system
├── ...
└── models
    ├── TD          # Text detection models.
    └── TR          # Text recognition models.
```
# Data
To finetune and benchmark models for handwritten text detection and recognition, IAM, CVL and a small constructed dataset is used.

Use the following file structure.
```
sc-hwn-system
├── ...
└── data
    ├── IAM 
        ├── aachen_splits
        ├── forms
        ├── lines
        ├── words
        └── xml
    └── system_dataset
        ├── images
        └── labels         
```
## IAM
The IAM database is used to finetune YOLOv5 and STAR-Net and for evaluation. It can be downloaded [here](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). For training and evaluation we use the [Aachen splits](data/IAM/aachen_splits) downloaded [here](https://www.openslr.org/56/). 

## CVL

## IMGUR5K

## System dataset
The [system dataset](data/system_dataset) is used in [benchmark.py](src/benchamrk.py) to evaluate the entire system. It was constructed to include some handwritten words with spelling errors.

# Running the application
```cmd
cd src 
python main.py
```
## Arguments
```--td_model [-d]```: Text detection model *{yolo, east}*, *default=yolo* <br>
```--tr_model [-r]```: Text recognition model *{resnet, dtrb}*, *default=dtrb* <br>
```--post_process [-p]```: Choose how to post process recognized text. *{trocr, symspell, None}*, *default=symspell*.<br>
```--spell_check [-s]```: Choose how to perform the spell correction.*{symspell}*, *default=symspell*<br>
```--video_source [-v]```: Video source, *default=0*. <br>

# Benchmarking of models
There are two notebooks available to evaluate individal models and one file available for evaluation of the entire system. More information about the benchmarks are available in [benchmark-notebook/README.md](benchmark-notebook/README.md).
## Text detection benchmark
The handwritten text detection models are evaluated using the notebook: [text-detection-benchmark.ipynb](benchmark-notebook/text-detection-benchmark.ipynb). The notebook runs the models on the IAM Forms (Aachen test split) dataset.

### Acquired results
|Model |Precision |Recall |Inference Time (ms) / word|
|---         |---   |---    |---   
|EAST        |0.677 |0.566 |3.966
|YOLOv5n     |0.959 |0.944 |0.814   
|YOLOv5s     |0.960 |0.954 |0.918   
|YOLOv5l     |0.962 |0.961  |1.075 

## Text recognition benchmark
The handwritten text recognition models are evaluated using the notebook:[text-recognition-benchmark.ipynb](benchmark-notebook/text-detection-benchmark.ipynb). The notebook runs the models on the IAM Words (Aachen test split) dataset.

### Acquired results
|Model |CER % |WER % |Inference Time (ms) / word|
|---                 |---   |---    |---   
|OrthoNet            |6.24  |18.72 |13.73
|ResNet_CTC          |35.00 |68.66 |55.33   
|DenseNet_CTC        |52.72 |86.71 |5.69   
|CRNN_VGG_BiLSTM_CTC |34.38 |44.52 |17.58  

## Full system and spell correction
To evaluate the entierty of the system and check how many spelling mistakes the system finds, run the file [benchmark.py](src/benchmar.py). <br>
```cmd
cd src
python benchmark.py
```
### Arguments
```--td_model [-d]```: Text detection model *{yolo, east}*, *default=yolo* <br>
```--tr_model [-r]```: Text recognition model *{resnet, dtrb}*, *default=dtrb* <br>
```--post_process [-p]```: Choose how to post process recognized text. *{trocr, symspell, None}*, *default=symspell*.<br>
```--spell_check [-s]```: Choose how to perform the spell correction.*{symspell}*, *default=symspell*<br>
```--data_path [-p]```: Path to the data to run the system on, *default='../data/system_dataset/'* <br>
```--max_samples [-m]```: Number of samples to test on. *default=None* <br>
```--output_name [-o]```: Name of the file where the benchmark results are stored. <br>

### Acquired results
Best results were acquired using the following configuration:
- Text Detection: YOLOv5s
- Text Recognition: OrthoNet
- Post Process: Symspell
- Spell Check: Symspell

CER (%)|F1 (%)| Recall (%) | Precision (%) |Total Inference Time (ms) / word|
|--- |--- |--- |---|---
5.92| 60.26| 68.68| 56.91| 16.656    

To calculate F1, recall and precision, the benchmark compares the true incorrect spelled words with the words that the system flags as incorrectly spelled.

# Finetuning YOLOv5 for handwritten text detection
The YOLOv5 model is finetuned for handwritten text detection using the IAM Forms dataset. Read the chapter [Data](#data) for how to download and place the dataset.
## Preparing the data
The data is prepared running the notebook: [IAM_yolo_preprocess.ipynb](data-preprocessing/IAM_yolo_preprocess.ipynb). Running the notebook creates a folder *data/IAM_yolo* containing the data needed to finetune the model. Upload the folder to the Google Drive and edit the *data.yaml* file to match your paths.
## Finetuning the model
The model is trained following the Google Colab notebook found [here](https://colab.research.google.com/drive/1rBJSTW-RZ9AzbR12BEjNkh58NkczuZky?usp=sharing).
# Acknowledgements
## Reference
## Contact
## License
