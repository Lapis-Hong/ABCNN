
# ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
TensorFlow implementation of **ABCNN**, proposed by [Wenpeng Yin et al.](https://arxiv.org/pdf/1512.05193.pdf)

I try best to write most elegant and effective model implementation. 

## Prerequisites

 - Python 2.7
 - TensorFlow >= 1.10 

## Datasets
- AS task with [WikiQA](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)
- PI task with [MSRP(Microsoft Research Paraphrase)](https://www.microsoft.com/en-us/download/details.aspx?id=52398)

## Models

## Extensions


## TODOs
add ABLSTMs
add metrics
fix bugs

## Results
TODO

### Notes:
This is slightly different with origin paper, here I do not use additional **LR** or 
**SVM** classifier, it's not elegant at all.


## Usage
> python train.py

> python predict.py
 
## References

- https://github.com/galsang/ABCNN 
- [Origin code](https://github.com/yinwenpeng/Answer_Selection/tree/master/src)
- [Wenpeng Yin et al.](https://arxiv.org/pdf/1512.05193.pdf)

