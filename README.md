## Chinese Text MultiClassification

Description:
	This project was created for text classification,as we all know, text classication played an importent role in nlp tasks,so it was worth studying with different
algorithms,and in this repo we will supply different methods to complete classification,include deep learning and traditional algorithms like svm, naive bayes.
The main features of this project that we develop a completed framework of classification from data pre-processing,feature selection, model training and predict,
and model update strategy,so that it can be easily migrated and used.

Table of Contents
=================

* [Chinese Text MultiClassification](chinese-text-multiclassification)
    * [Traditional Approach](#traditional-methods)
        * [Feature Selection && Extraction](#feature-selection-extraction)
        * [Text Classification](#text-classification)
    * [Deep Learning Approach](#deep-learning-methods)
        * [Word Embedding](#word-embedding)
        * [Text Classification](#text-classification)
            * [fastText](fastText)

## Traditional Approach
TBD
### Feature Selection && Extraction
TBD

## Deep Learning Approach
### Word Embedding

### Text Classification
As the deep learning algorithms develops, we can build a noval text classification easily instead of doing much more feature engineering,bellow I will give some common 
methods or algorithms to build a classification system,which can achive the state-of-art preformance.

#### fastText
[fastText](https://github.com/facebookresearch/fastText) is an useful library designed by Facebook team, which can be used for word representations and sentence 
classification.In this post I had create a vanillia demo based on [pyfasttext](https://github.com/vrasneur/pyfasttext) a python package of fastText.

##### Dependent
    - [pyfasttext], you can install it like

    ```bash
    pip install pyfasttext
    ```

##### Enjoy Demo
```bash
bash run.sh
```
In this bash, fristly it will be download a toy classification data created by myself, second it will train a classification model based on those data, finally it will
predict the samples from valid data and write the results into a new file in result.The data contains three labels, every label in each sample start with '__label__' 
which is the default label format as raw fastText, you also can set your own label format by apis.


