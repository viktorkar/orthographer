
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
The [system dataset](data/system_dataset) is used in [benchmark.py](src/benchmark.py) to evaluate the entire system. It was constructed to include some handwritten words with spelling errors.

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
There are two notebooks available to evaluate individual models and one file available for evaluation of the entire system. More information about the benchmarks are available in [benchmark-notebook/README.md](benchmark-notebook/README.md).
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
To evaluate the entirety of the system and check how many spelling mistakes the system finds, run the file [benchmark.py](src/benchmark.py). <br>
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
We would like to thank Haute Ecole d'Ingénierie et de Gestion du Canton de Vaud (HEIG-VD) for providing us with the opportunity to realize this project. Particularly, we want to recognize our supervisors Pierre Nugues at Lund University and Marcos Rubinstein at HEIG-VD, who contributed with vital suggestions, feedback and assistance during the developmental process, in addition to input on the writing of the thesis.

## Citation
Please consider citing this work in your publications if it helps your research.

```
@mastersthesis{vkaa2022,
    author = {Viktor Karlsson and Aston Åkerman},
    title = {Real-time detection of spelling mistakes in handwritten notes},
    school = {Lund University, LTH},
    year = 2022
}
```

## Reference
For full reference list, see the [thesis]()

```
@inproceedings{baek2019STRcomparisons,
title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
booktitle = {International Conference on Computer Vision (ICCV)},
year={2019},
pubstate={published},
tppubtype={inproceedings}
}

@misc{symspell,
    author = {SeekStorm},
    title = {SymSpell},
    year = {2021},
    note = {},
    url = {https://github.com/wolfgarbe/SymSpell},
    publisher: = {SeekStorm}
}
```

## Contact
Feel free to contact us if there are questions about the code or thesis:

[Aston Åkerman](https://www.linkedin.com/in/astonakerman/) &
[Viktor Karlsson](https://www.linkedin.com/in/viktor-karlsson-b7209317a/)
## License
```
Copyright 2022-present Aston Åkerman & Viktor Karlsson
```

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
```
http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### Symspell
```
Copyright (c) 2022 Wolf Garbe
Version: 6.7.2
Author: Wolf Garbe <wolf.garbe@seekstorm.com>
Maintainer: Wolf Garbe <wolf.garbe@seekstorm.com>
URL: https://github.com/wolfgarbe/symspell
Description: https://seekstorm.com/blog/1000x-spelling-correction/

MIT License

Copyright (c) 2022 Wolf Garbe

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

https://opensource.org/licenses/MIT
```