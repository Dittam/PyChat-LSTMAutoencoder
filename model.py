from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.initializers import Constant
from keras.utils import plot_model
from tensorflow import placeholder
from keras.preprocessing.sequence import pad_sequences
import preprocessing
import wordEmbeddings
import numpy as np


def str2tokens(sentence, word2idx, MAX_SENTENCE_LENGTH):
  words = sentence.lower().split()
  tokensList = list()
  for word in words:
    try:
      tokensList.append(word2idx[word])
    except:
      tokensList.append(word2idx["<UNK>"])
  return pad_sequences([tokensList], maxlen=MAX_SENTENCE_LENGTH, padding='post')


def defineModels(MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE, LATENT_DIM,
                 EMBEDDING_DIM, embeddingMatrix):

  wordEmbeddingLayer = Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, embeddings_initializer=Constant(
    embeddingMatrix), input_length=MAX_SENTENCE_LENGTH, trainable=False, name="wordEmbeddingLayer")

  # ----TRAINING MODEL----

  # ---ENCODER---
  encInputLayer = Input(shape=(MAX_SENTENCE_LENGTH,), name="encoderInputLayer")
  encEmbeddedWords = wordEmbeddingLayer(encInputLayer)
  encLSTM1 = LSTM(LATENT_DIM, return_state=True, name="encoderLSTM1")
  _, hState, cState = encLSTM1(encEmbeddedWords)
  latentEncoding = [hState, cState]

  # ---DECODER---
  decInputLayer = Input(shape=(MAX_SENTENCE_LENGTH,), name="decoderInputLayer")
  decEmbeddedWords = wordEmbeddingLayer(decInputLayer)
  decLSTM1 = LSTM(LATENT_DIM, return_sequences=True, return_state=True,
                  name="decoderLSTM1")
  decHiddenStates, _, _ = decLSTM1(
    decEmbeddedWords, initial_state=latentEncoding)
  decDense = Dense(MAX_VOCAB_SIZE, activation='softmax',
                   name="decoderDenseLayer")
  decOutput = decDense(decHiddenStates)

  trainModel = Model([encInputLayer, decInputLayer], decOutput)
  trainModel.compile(optimizer='rmsprop',
                     loss='categorical_crossentropy', metrics=['accuracy'])

  # ----INFERENCE MODEL----

  # ---ENCODER---
  encInferenceModel = Model(encInputLayer, latentEncoding)

  # ---DECODER---
  decHStateInput = Input(shape=(LATENT_DIM,), name="decoderHStateInput")
  decCStateInput = Input(shape=(LATENT_DIM,), name="decoderCStateInput")

  infDecHiddenStates, infStateH, infStateC = decLSTM1(
    decEmbeddedWords, initial_state=[decHStateInput, decCStateInput])
  decOutput = decDense(infDecHiddenStates)
  decInferenceModel = Model([decInputLayer, decHStateInput, decCStateInput], [
      decOutput, infStateH, infStateC])

  return trainModel, encInferenceModel, decInferenceModel


def runInference(sentence, eModel, dModel, word2idx, MAX_SENTENCE_LENGTH):
  idx2word = {v: k for k, v in word2idx.items()}
  stateValues = eModel.predict(str2tokens(
    sentence, word2idx, MAX_SENTENCE_LENGTH))
  targetSeq = np.zeros((1, 20))
  targetSeq[0, 0] = word2idx['<SOS>']
  eos = word2idx['<EOS>']
  outputSentence = []

  for _ in range(MAX_SENTENCE_LENGTH):
    output_tokens, h, c = dModel.predict([targetSeq] + stateValues)
    idx = np.argmax(output_tokens[0, 0, :])
    if eos == idx:
      outputSentence.append("<EOS>")
      break
    if idx > 0:
      outputSentence.append(idx2word[idx])

    targetSeq[0, 0] = idx
    stateValues = [h, c]

  return ' '.join(outputSentence)


if __name__ == '__main__':
  import numpy as np
  import wordEmbeddings
  import preprocessing

  MAX_SENTENCE_LENGTH = 20
  MAX_VOCAB_SIZE = 10000
  LATENT_DIM = 250
  EMBEDDING_DIM = 100

  # inputs, targets, word2idx = preprocessing.tokenizeData(
  #   MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE)
  # embeddingMatrix = wordEmbeddings.generateEmbeddingMatrix(
  #   "glove.6B.100d.txt", word2idx, MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE, EMBEDDING_DIM, True)

  embeddingMatrix = np.load("embeddingMatrixGolve6b100.npy")

  trianModel, eModel, dModel = trainModel(MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE,
                                          LATENT_DIM, EMBEDDING_DIM, embeddingMatrix)

  # plot_model(model, to_file='modelPlot.png',
  #            show_shapes=True, show_layer_names=True)
