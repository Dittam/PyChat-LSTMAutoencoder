from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from preprocessing import CornellMovieDataGenerator
import numpy as np


def str2tokens(sentence, word2idx, MAX_SENTENCE_LENGTH):
  words = sentence.split()
  tokensList = list()
  for word in words:
    try:
      tokensList.append(word2idx[word])
    except:
      tokensList.append(word2idx["<UNK>"])
  return pad_sequences([tokensList], maxlen=MAX_SENTENCE_LENGTH, padding='post')


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
      break
    if idx > 0:
      outputSentence.append(idx2word[idx])

    targetSeq[0, 0] = idx
    stateValues = [h, c]

  return ' '.join(outputSentence)


if __name__ == '__main__':

  MAX_SENTENCE_LENGTH = 20
  MAX_VOCAB_SIZE = 10000
  LATENT_DIM = 250
  EMBEDDING_DIM = 100
  BATCH_SIZE = 540
  EPOCHS = 20
  sentence = "hello how are you"

  dataGenerator = CornellMovieDataGenerator(
    'data/trainInputs.txt', 'trainTargets.txt', BATCH_SIZE, MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE)
  trainModel = load_model(
      "trainedModels/trainModel_20EPOCHS_ 2.63LOSS.h5")
  eModel = load_model(
    "trainedModels/eModel_20EPOCHS_ 2.63LOSS.h5", compile=False)
  dModel = load_model(
    "C:/trainedModels/dModel_20EPOCHS_ 2.63LOSS.h5", compile=False)

  print(runInference(sentence, eModel, dModel,
                     dataGenerator.word2idx, MAX_SENTENCE_LENGTH))
