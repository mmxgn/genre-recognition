# Music Genre Classification
Build a machine learning (ML) model that takes audio files as input and returns a corresponding music genre.

## Training the model
Model training is done using the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset for Music Genre Classification (MCG). This dataset consists of 1000 `.wav` (22KHz 16bit mono) files of 30 second audio excerpts   spanning 10 musical genres. Each genre consists of 100 tracks.

### Preliminaries
It is recommended you install the requirements and you run the code in the notebooks in a separate virtual environment:

```
$ virtualenv venv
$ source venv/bin/activate
```

### Requirements

The code was written on Python 3.7.4. The classifier relies on [MusiCNN](https://github.com/jordipons/musicnn) for computing musically-relevant feature vectors, [Scikit-learn](https://scikit-learn.org/stable/index.html) for training the final classifier, [Pandas](https://pandas.pydata.org/) and [Seaborn](https://seaborn.pydata.org/) for visualising the results. 

You can install most of the requirements with:

```bash
$ pip install -r requirements.txt
```

The only exception is `MusiCNN` which you need to install via the command line:

```bash
$ git clone https://github.com/jordipons/musicnn
$ cd musicnn
$ python setup.py install
```

### Fetching and extracting the data 

Visit the [GTZAN](http://marsyas.info/downloads/datasets.html) homepage and download the GTZAN genre collection (approx. 1.2GB). Extract `genres.tar.gz` to the `data` folder in this repo:

```
tar xzf /path/to/genres.tar.gz -C data/ 
```

### Training and Evaluation 


|Classifier                    |Avg. Acc| Avg. P| ROC-AUC| Time (training) | Time (testing) |
|------------------------------|--------|-------|--------|----   | ----|
|SVM(RBF,C=1)<sup>1</sup>|0.7724   | --    |  --    |      ---     |--- |
|SVM(RBF,C=1)  |         0.7860   | 0.8822      | 0.9740       |4m40s| 45.9s|
|XGBoost       |         0.7826   |  0.8861     | 0.9707       |26.2s| 569ms |


Please see the jupyter notebooks in the root directory of this repo for how the models are trained and evaluated.


<sup>1</sup>Reported at https://github.com/jordipons/sklearn-audio-transfer-learning.

### Classifying from within a python script or the command line 

The classifier is also provided as a python function in `python/classify.py`. You can use it to classify an arbritrary audio file as follows:

```bash
# Switch to directory 
$ cd python/

# (optional) run unit tests
$ python test_classify.py

# python classify.py ../data/genres/blues/blues.00002.wav
blues$
```

The script does not add a new line after execution in order allow its output to be used in e.g. shell scripts. You can also use it from inside
another python script as such:

```python
import classify
classify.classify_audio(
    "../data/genres/blues/blues.00002.wav",
    "../models/features_classifier.pkl",
)
```

The first argument is the audio to be classified, and the second the classifier used (see the notebook files in this repo).

## Deploying with `docker-compose`

You can find a flask microservice for the classifier in the `docker` directory. First you need
to place `features_classifier.pkl` from `models/` or Releases in the `docker/classifier/models` directory. Then you can build and run the microservice with:

```bash
$ cd docker
$ docker-compose up --build -d
```

You can then use `curl` in order to classify audio files. E.g.:

```bash
curl -F 'audio=@classical.00067.wav' http://localhost:5000
{"duration":0.9515135288238525,"message":"classical","status":"success"}
```

When you finish testing you can shut down the container with:
```bash
$ docker-compose down
```
## Speeding up deployment

In order to reduce the time taken for classification, as well guarantee constant time regardless of song duration, we used the `sox` library to trim 7 seconds around the middle of the song (which translates to roughly 5 feature vectors). This resulted in a constant duration of ~0.95 seconds per song (~2-fold decrease) regardless of song duration while accuracy suffered by a mere 2.4%.

|Type|Time (ms) |Accuracy|
|----|----|--------|
|Original|1963±44|0.7828|
|Truncated (7sec)|998±37|0.7586|
|Difference| 197%| ~2.4%| 

# Technical Report

Please see `doc/TechReport.pdf` for a more comprehensive technical report.
