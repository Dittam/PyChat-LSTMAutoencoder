from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from preprocessing import CornellMovieDataGenerator
from model import runInference, str2tokens
import numpy as np

MAX_SENTENCE_LENGTH = 20
MAX_VOCAB_SIZE = 10000
LATENT_DIM = 250
EMBEDDING_DIM = 100
BATCH_SIZE = 540
EPOCHS = 20
sentence = "hello how are you"

dataGenerator = CornellMovieDataGenerator(
  'data/trainInputs.txt', 'data/trainTargets.txt', BATCH_SIZE, MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE)
trainModel = load_model("trainedModels/trainModel_20EPOCHS_ 2.63LOSS.h5")
eModel = load_model(
  "trainedModels/eModel_20EPOCHS_ 2.63LOSS.h5", compile=False)
dModel = load_model(
  "trainedModels/dModel_20EPOCHS_ 2.63LOSS.h5", compile=False)

print(runInference(sentence, eModel, dModel,
                   dataGenerator.word2idx, MAX_SENTENCE_LENGTH))
