from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from preprocessing import CustomDataGenerator
from model import defineModels
from keras.callbacks import LambdaCallback, ReduceLROnPlateau
import numpy as np

MAX_SENTENCE_LENGTH = 20
MAX_VOCAB_SIZE = 1879
LATENT_DIM = 250
EMBEDDING_DIM = 100
BATCH_SIZE = 25
EPOCHS = 2
LR_DECAY = 0.2
path = "C:/Users/ditta/Desktop/kagglePyBotDataset"


def on_epoch_end(epoch, logs):
  samples = ["hello how are you", "is it a nice day today", "why not"]
  for i in samples:
    print(runInference(i, eModel, dModel,
                       dataGenerator.word2idx, MAX_SENTENCE_LENGTH))
  print("")
  if epoch % 5 == 0:
    eModel.save(
      path + "trainedModels/eModel{}_{}.h5".format(epoch, logs["loss"]))
    dModel.save(
      path + "trainedModels/dModel{}_{}.h5".format(epoch, logs["loss"]))


customCallback = LambdaCallback(on_epoch_end=on_epoch_end)
reduceLRCallback = ReduceLROnPlateau(
    monitor='loss', factor=LR_DECAY, patience=1, min_lr=0.0001)
callbackLst = [customCallback, reduceLRCallback]


dataGenerator = CustomDataGenerator(
    path + '/trainInputs.txt', path + '/trainTargets.txt', BATCH_SIZE, MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE)
embeddingMatrix = np.load(path + "/embeddingMatrixGolve6b100_2.npy")
print("training...")
trainModel, eModel, dModel = defineModels(
  MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE, LATENT_DIM, EMBEDDING_DIM, embeddingMatrix)

trainHist = trainModel.fit_generator(generator=dataGenerator, epochs=EPOCHS,
                                     use_multiprocessing=True, workers=4,
                                     callbacks=callbackLst)

# plot_model(trainModel, show_shapes=True, show_layer_names=True,to_file='trainingModel.png')

# trainModel.save( 'trainModel.h5' )
# eModel.save("eModel.h5")
# dModel.save("dModel.h5")
