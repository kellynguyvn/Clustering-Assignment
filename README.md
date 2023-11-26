# Clustering-Assignment
## Goals
Implement various clustering algorithms in colabs - use generously gpt4 latest. Submit one colab per assignment - provide complete colab with proper documentation etc.,. 

## Clustering Algorithms
### a) K-Means clustering from scratch
#### Imports
```
```
### References
https://colab.sandbox.google.com/github/SANTOSHMAHER/Machine-Learning-Algorithams/blob/master/K_Means_algorithm_using_Python_from_scratch_.ipynb
https://colab.sandbox.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.11-K-Means.ipynb
https://developers.google.com/machine-learning/clustering/programming-exercise
https://colab.sandbox.google.com/github/google/eng-edu/blob/main/ml/clustering/clustering-supervised-similarity.ipynb?utm_source=ss-clustering&utm_campaign=colabexternal&utm_medium=referral&utm_content=clustering-supervised-similarity#scrollTo=eExms-TP8Hn6

 

### b) Hierarchical clustering (not from scratch)
#### Imports
```
```
### References
https://colab.sandbox.google.com/github/saskeli/data-analysis-with-python-summer-2019/blob/master/clustering.ipynb

### c) Gaussian mixture models clustering (not from scratch)
#### Imports
```
```
### References
https://colab.sandbox.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb

 

### d) DB Scan clustering (not from scratch) using pycaret library 
#### Imports
```
!pip install pycaret
```
### References
https://pycaret.org/create-model/ 
https://towardsdatascience.com/clustering-made-easy-with-pycaret-656316c0b080
http://www.pycaret.org/tutorials/html/CLU101.html

### e) Demonstrate anomaly detection using pyOD using any usecase
eg: univariate or multivariate 
#### Imports
```
!pip install pyod
!pip install pandas numpy matplotlib scikit-learn
```
### References 
https://neptune.ai/blog/anomaly-detection-in-time-series 
https://github.com/ritvikmath/Time-Series-Analysis/blob/master/Anomaly%20Detection.ipynb 

 

### f) Illustrate clustering of timeseries data using pretrained models 
#### Imports
```
!pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tslearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import tensorflow as tf
```
### References
https://github.com/V-MalM/Stock-Clustering-and-Prediction
https://github.com/qianlima-lab/time-series-ptms
https://github.com/effa/time-series-clustering
https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM
https://github.com/cure-lab/Awesome-time-series#time-series-clustering
https://github.com/qingsongedu/awesome-AI-for-time-series-papers

 

### g) Write a colab to illustrate clustering  of documents. use state of art embeddings (LLM Embeddings).
#### Imports
```
```
### References
https://github.com/simonw/llm-cluster
https://simonwillison.net/2023/Sep/4/llm-embeddings/#llm-cluster
https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/clustering
https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
https://github.com/Ruchi2507/Text-Clustering

 

### h) Write a colab for clustering with images using imagebind llm embeddings 
#### Imports
```
!pip install tensorflow
!pip install scikit-learn
!pip install matplotlib
!pip install pillow

```
### References
https://medium.com/@tatsuromurata317/image-bind-metaai-on-google-colab-free-843f30a4977c
https://towardsdatascience.com/introduction-to-embedding-clustering-and-similarity-11dd80b00061 
https://cobusgreyling.medium.com/using-openai-embeddings-for-search-clustering-83840e971e97 

### i) Write a colab for audio embeddings using imagebind llms
#### Imports
```
```
### References
https://towardsdatascience.com/k-means-clustering-and-pca-to-categorize-music-by-similar-audio-features-df09c93e8b64
https://mct-master.github.io/machine-learning/2023/04/25/ninojak-clustering-audio.html 
https://ridakhan5.medium.com/audio-clustering-with-deep-learning-a7991d605fa5 
https://www.kaggle.com/code/humblediscipulus/audio-feature-extraction-and-clustering

 

