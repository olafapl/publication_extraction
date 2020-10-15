# CNN-Sentence

Tensorflow/Keras implementation of a CNN for sentence classification, based on the paper ["
Convolutional Neural Networks for Sentence Classification"](https://arxiv.org/abs/1408.5882) by Yoon
Kim.

This model uses pretrained [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/), and is
trained on the [HomePub dataset](https://people.eng.unimelb.edu.au/zr/data/homepub.html). [This
article](https://keras.io/examples/nlp/pretrained_word_embeddings/) from the Keras docs was helpful
for getting started with pre-trained embeddings.

## Training

Run `main.py`. The script assumes that GloVe word embeddings are located in `data/glove/` and that
the HomePub dataset is located in `data/homepub-2500/`.

## Other Implementations

- [CNN_sentence](https://github.com/yoonkim/CNN_sentence) by Yoon Kim.
- [textClassifier](textClassifier).
- [CNN-for-Sentence-Classification-in-Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras).
