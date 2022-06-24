
# Ortographer: Real-Time Detection of Spelling Mistakes in Handwritten Notes. <!-- omit in toc --> 

Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [Additional optional dependencies](#additional-optional-dependencies)
- [Models](#models)
- [Data](#data)
  - [IAM](#iam)
  - [CVL](#cvl)
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
- [Finetuning STAR-Net for handwritten text recognition](#finetuning-star-net-for-handwritten-text-recognition)
  - [Preparing the data](#preparing-the-data-1)
  - [Finetuning the model](#finetuning-the-model-1)
- [Acknowledgements](#acknowledgements)
  - [Reference](#reference)
  - [Contact](#contact)
  - [License](#license)

# Introduction 
Ortographer is an end-to-end application for real-time detection of spelling mistakes in handwritten notes. The system was created as a part of [this]() master thesis project in 2022.

To perform the task, the system runs a pipeline consisting of modules for video capturing, text detection, text recognition, post processing and finally spell correction. The pipeline is displayed in the image below.

 ![System Architecture](/misc/sys_arch.png)

# Installation
We recommend using [Anaconda](https://www.anaconda.com/distribution/) or the light-weight version [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to make sure that the correct dependencies are installed. Once installed, create an environment by running the following command. The environment will contain all libraries needed to run the program with default settings. By default, the environment will be named `ortographer`, but you can choose another name using the `-n` option.

**OBS**. If running only on CPU, edit environment.yml to avoid installing GPU dependencies.

    $ conda env create -f environment.yml

Next, activate the environment:

    $ conda activate ortographer

## Additional optional dependencies
**neuspell**<br>
If you wish to run the program with the flag *--post_process neuspell*, you also need to install neuspell following [these](https://pypi.org/project/neuspell/) steps. We had problems downloading their pretrained models using the methods provided by their library but was able to do it manually using their [google drive folder](https://drive.google.com/drive/folders/1jgNpYe4TVSF4mMBVtFh4QfB2GovNPdh7?usp=sharing). Using neuspell in our system requires the **subwordbert-probwordnoise** pretrained model.

# Models
The system uses trained machine learning models for handwritten text detection and handwritten text recognition. All finetuned models can be found in the [google drive folder here](https://drive.google.com/drive/folders/1jRYy1_Cs7pLex1zoNjcQNYo363Kt2Mt6?usp=sharing). Acquired performances of the models are given under [Benchmarking of models](#benchmarking-of-models) and more information can be found in [models/README.md](models/README.md).

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
TODO

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
```--post_process [-p]```: Choose how to post process recognized text. *{trocr, spellcheck, neuspell, None}*, *default=spellcheck*.<br>
```--spell_check [-s]```: Choose how to perform the spell correction.*{spellcheck}*, *default=spellcheck*<br>
```--video_source [-v]```: Video source *{0=webcam, 1=external}*, *default=0*. <br>

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
|STAR-Net            |8.39  |23.95 |13.73
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
```--post_process [-p]```: Choose how to post process recognized text. *{trocr, spellcheck, neuspell, None}*, *default=spellcheck*.<br>
```--data_path [-p]```: Path to the data to run the system on, *default='../data/system_dataset/'* <br>
```--max_samples [-m]```: Number of samples to test on. *default=None* <br>
```--output_name [-o]```: Name of the file where the benchmark results are stored. <br>

### Acquired results
|Detection |Recognition|Post Process|CER (%)|F1 (%)| Precision (%) |Recall (%) |Inference (ms) / word|
|---            |---   |---    |--- |--- |--- |---|---
|YOLOv5s        |STAR-Net |Spellcheck |9.126 |47.932 |51.772 |53.535 |-     
|YOLOv5s        |STAR-Net |TrOCR      |7.973 |47.842 |48.610 |56.772|-    
|YOLOv5s        |STAR-Net |Neuspell   |9.162 |48.095 |51.868 |54.045 |-    
|YOLOv5s        |STAR-Net |None   |9.01 |41.599 |35.415 |60.798 |-

# Finetuning YOLOv5 for handwritten text detection
The YOLOv5 model is finetuned for handwritten text detection using the IAM Forms dataset. Read the chapter [Data](#data) for how to download and place the dataset.
## Preparing the data
The data is prepared running the notebook: [IAM_yolo_preprocess.ipynb](data-preprocessing/IAM_yolo_preprocess.ipynb). Running the notebook creates a folder *data/IAM_yolo* containing the data needed to finetune the model. Upload the folder to the Google Drive and edit the *data.yaml* file to match your paths.
## Finetuning the model
The model is trained following the Google Colab notebook found [here](https://colab.research.google.com/drive/1rBJSTW-RZ9AzbR12BEjNkh58NkczuZky?usp=sharing).

# Finetuning STAR-Net for handwritten text recognition
TODO
## Preparing the data
TODO
## Finetuning the model
TODO
# Acknowledgements
TODO
## Reference
TODO
## Contact
TODO
## License
TODO
