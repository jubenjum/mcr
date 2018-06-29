# Monkey Call Recognition & Features extraction
[![Build Status](https://travis-ci.org/primatelang/mcr.svg?branch=master)](https://travis-ci.org/primatelang/mcr)
[![Anaconda-Server Badge](https://anaconda.org/primatelang/mcr/badges/installer/conda.svg)](https://conda.anaconda.org/primatelang)

Python code for training, decoding, and evaluating an automatic monkey call recognizer. 
Support of feature extraction and dimension reduction


## Features

* Classification and decoding of calls in isolation
* Automatic transcription of audio files
* Feature extraction and dimension reduction
* Parsing [Praat](http://www.fon.hum.uva.nl/praat/) TextGrid files


## Installation

For installing the mcr package you will require a python 2.7 distribution in your system. 
You can install *mcr* by selecting one of the following methods: 

### Method 1: virtualenv + pip + setup
Install python 2.7.x using your system's package manager. First you will need to 
get this package:

    $ git clone https://github.com/primatelang/mcr

Optionally, install virtualenv to keep separate the required packages and then do:

    $ virtualenv -p $(which python2.7) ~/mcr_venv && source ~/mcr_venv/bin/activate

Next, install the required packages:

    $ cd mcr
    $ pip install -r requirements.txt

Finally run the installation script:

    $ python setup.py install

And you're good to go.

### Method 2 : anaconda
Follow the instructions from [anaconda webpage](https://www.anaconda.com/download/) 
to install a suitable distribution for your system. The supported architectures of
this module are Linux and OSX. Once anaconda installed on your system, you can
follow these commands:

    $ conda create --name monkeys
    $ source activate monkeys
    $ conda install -c primatelang mcr

it will install scripts and all the requirements for *mcr*. 


# Usage

The system consists of four sets of scripts with similar interfaces, for
*segmented* recognition and *transcription*, *feature* extraction and
*dimension reduction* respectively. all scripts can be found in *bin* and *mcr*
folders.  We will discuss these scripts below. First, we need to discuss the
annotation and configuration formats.


## Annotation files

The *train*, *predict*, *eval* and *feature extraction* scripts all require an
annotation file. The annotation file is a csv-file with **filename**, **start**
and **end** columns and an optional **label** column which is needed only for
the *train* and *eval* and *feature extraction* scripts. The rows in the
annotation file point to segments of audio files, denoted by a filename, start
and end times and possible call label. As an example, suppose we have two
one-second audio recordings, */home/me/A.wav* and */home/me/B.wav* and we have
annotated a *PYOW* call in file *A* at the time interval *0.100 - 0.200* and
a *HACK* call in file *B* at the time interval *0.500 - 0.600*; we then make
an annotation file (*annotation.csv*) with the following contents to train our
classifier or to do feature extraction:

```
filename,start,end,label
/home/me/A.wav,0.000,0.100,SIL
/home/me/A.wav,0.100,0.200,PYOW
/home/me/A.wav,0.200,1.000,SIL
/home/me/B.wav,0.000,0.500,SIL
/home/me/B.wav,0.500,0.600,HACK
/home/me/B.wav,0.600,1.000,SIL
```

where we used *SIL* to denote the absence of calls, annotations that you can
skip if you are only doing feature extraction. Note that, although we have
two audio files, we only make a single annotation file for the classifier
script. In case we have a trained classifier available and wish to use it to
transcribe a new one-second recording *C.wav* we have made and haven't manually
annotated, we can make an annotation file without a **label** column:

```
filename,start,end
/home/me/C.wav,0.000,1.000
```

## Configuration files

The *train* and *feature extraction* scripts require a configuration file which
lists the settings for the feature extractors and the hyper-parameters for the
classifiers. This file needs to be in *toml* format. Examples of these files
with default settings can be found in *config/segmented.cfg* and
*config/transcription.cfg*. These example files list all the parameters to the
system, so if you wish to play around with different settings, simply changing
these configuration files is a good option.

To configure the *dimension reduction* algorithms you will need also the file
*mcr/algorithms.json*, in  *json* format; you can also copy that 
file to your working directory to keep a record of the parameters you used 
when running the *dimension reduction* scripts.

## Segmented classification

*Segmented classification* refers to classifying individual monkey calls in
isolation, i.e. we know their exact location in an audio stream and need to
learn their identity. The following scripts are supplied:

- *bin/segmented_train.py*
- *bin/segmented_predict.py*
- *bin/segmented_eval.py*

For example, to train a classifier using the above annotated calls, we do:

  $ python segmented_train.py annotation.csv segmented.cfg trained.clf

Note that this will only consider the annotated *calls*, the script will ignore
any non-call (*SIL*) segments in the audio files.

To use this trained classifier to predict the annotations again, we do:

  $ python segmented_predict.py annotation.csv trained.clf predicted.csv

To compare the predicted labels to the annotation, we do:

  $ python segmented_eval.py annotation.csv predicted.csv

Which will print scores to the standard output.


## Transcription

*Transcription* refers to the automatically transcribing entire audio files,
marking where calls occur and their identity and where there is only silence
(or, more likely, background noise). The following scripts implement a complete
system for training, decoding and evaluating a transcription classifier:

- *bin/transcription_train.py*
- *bin/transcription_predict.py*
- *bin/transcription_eval.py*

The scripts for complete transcription of audio files follow the same
conventions as those above. They will however take considerably longer to run.

## Feature extraction 

A script is available to extract features from audio files, to run it you
should have the *toml* configuration file, see "Configuration files" section.
For example, to extract features from the *annotations.csv* file, you can do:

    $ extract_features annotations.csv extraction.cfg -o features.csv

The output is in the *features.csv* file, the output is a csv-file that contains in
the first column the *label* and the rest of columns are the flattened features.

The option *stacksize* in the configuration file controls the length of the features,
if it is set to *0*, the script will extract all available frames from the sample and 
defined by the time-stamps for the sample in the *annotations.csv* file. Any other
value will extract the selected number of frames.


## Features dimension reduction

An script used to reduce the dimension of the features, you can use it to
discover relations between different calls. It takes the output from the
*feature extraction* script; eight reduction methods are available: Principal
Component Analysis (PCA), Linear Discriminant Analysis (LDA), Latent Semantic
Analysis (LSA), t-distributed Stochastic Neighbor Embedding (t-SNE), neuronal
network -NN- auto-encoder (AE) auto-encoder recurrent NN (LSTM), RNN triplet
loss with memory (TRIPLETLOSS) and RNN embeddings (LSTMEMBED), the first four
works for fix size features, and the rest can be process fix or variable size
features.

For example to reduce features from the output data in "Feature extraction" section,
and using the RNN auto-encoder (LSTM): 

    $ reduce_features features.csv extraction.cfg -r LSTM -o lstm.csv 

The output is *lstm.csv* file, the output is a csv-file that contains in
the first column the *label* and the rest of columns are the shrinked features.

For more information about how to use the scripts, see their help messages.
