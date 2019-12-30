from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from preprocessing import CornellMovieDataGenerator
from model import defineModels
import numpy as np

MAX_SENTENCE_LENGTH = 20
MAX_VOCAB_SIZE = 10000
LATENT_DIM = 250
EMBEDDING_DIM = 100
BATCH_SIZE = 540
EPOCHS = 20

dataGenerator = CornellMovieDataGenerator(
    'data/trainInputs.txt', 'data/trainTargets.txt', BATCH_SIZE, MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE)
embeddingMatrix = np.load("data/embeddingMatrixGolve6b100.npy")

trainModel, eModel, dModel = defineModels(
  MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE, LATENT_DIM, EMBEDDING_DIM, embeddingMatrix)

trainHist = trainModel.fit_generator(generator=dataGenerator, epochs=EPOCHS,
                                     use_multiprocessing=True, workers=4)

trainModel.save('trainModel.h5')
eModel.save("eModel.h5")
dModel.save("dModel.h5")
