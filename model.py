from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.initializers import Constant
from keras.utils import plot_model
import numpy as np


def defineModels(MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE, LATENT_DIM,
                 EMBEDDING_DIM, embeddingMatrix):

  wordEmbeddingLayer = Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, embeddings_initializer=Constant(
    embeddingMatrix), mask_zero=True, input_length=MAX_SENTENCE_LENGTH, trainable=False, name="wordEmbeddingLayer")

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
  trainModel.compile(optimizer="rmsprop",
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


if __name__ == '__main__':
  MAX_SENTENCE_LENGTH = 20
  MAX_VOCAB_SIZE = 10000
  LATENT_DIM = 250
  EMBEDDING_DIM = 100
  LR = 0.001

  embeddingMatrix = np.load(
    "data/embeddingMatrixGolve6b100.npy")

  trianModel, eModel, dModel = defineModels(MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE,
                                            LATENT_DIM, EMBEDDING_DIM, embeddingMatrix)

  # plot_model(trianModel, to_file='modelPlot.png',
  #            show_shapes=True, show_layer_names=True)
