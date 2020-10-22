# CNN-Sentence

Tensorflow/Keras implementation of a CNN for sentence classification, based on the paper ["
Convolutional Neural Networks for Sentence Classification"](https://arxiv.org/abs/1408.5882) by Yoon
Kim.

This model uses pretrained [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/), and is
trained on the [HomePub dataset](https://people.eng.unimelb.edu.au/zr/data/homepub.html). [This
article](https://keras.io/examples/nlp/pretrained_word_embeddings/) from the Keras docs was helpful
for getting started with pre-trained embeddings.

## Differences from Model Described in Paper

- Because the publication string classification problem is a two-class classification problem, the
  output layer uses sigmoid activation instead of softmax.
- 300-dimensional GloVe embeddings are used instead of 300-dimensional word2vec embeddings.
- The Adam optimizer is used instead of Adadelta.

## Prerequisites

- Pre-trained GloVe word embeddings must be located in `data/glove/`.
- The HomePub dataset must be located in `data/homepub-2500/`.

The `glove.sh` and `homepub.sh` scripts can be used to download these.

## Training

Run `src/cnn_sentence.py`.

## Other Implementations

- [CNN_sentence](https://github.com/yoonkim/CNN_sentence) by Yoon Kim.
- [textClassifier](textClassifier).
- [CNN-for-Sentence-Classification-in-Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras).
