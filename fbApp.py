from fbchat import Client, ThreadType, Message
import asyncio
from keras.models import load_model
from preprocessing import clean
from model import runInference
import numpy as np
import pickle


def getResponse(text):
  return runInference(clean(text.split(' ', 2)[2]), eModel, dModel,
                      word2idx, MAX_SENTENCE_LENGTH)


class EchoBot(Client):
  async def on_message(self, mid=None, author_id=None, message_object=None, thread_id=None,
                       thread_type=ThreadType.USER, at=None, metadata=None, msg=None):
    await self.mark_as_delivered(thread_id, message_object.uid)
    await self.mark_as_read(thread_id)

    # If you're not the author, echo
    if author_id != self.uid:
      predictedResponse = getResponse(message_object.text)
      print(predictedResponse)
      await self.send(Message(text=predictedResponse), thread_id=thread_id, thread_type=thread_type)


def load_obj(name):
  with open(name + '.pkl', 'rb') as f:
    return pickle.load(f)


MAX_SENTENCE_LENGTH = 20
MAX_VOCAB_SIZE = 10000
LATENT_DIM = 250
EMBEDDING_DIM = 100
BATCH_SIZE = 540
EPOCHS = 20

word2idx = load_obj('data/word2idx')
# dataGenerator = CornellMovieDataGenerator(
#   'data/trainInputs.txt', 'data/trainTargets.txt', BATCH_SIZE, MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE)
trainModel = load_model("trainedModels/trainModel_20EPOCHS_ 2.63LOSS.h5")
eModel = load_model(
    "trainedModels/eModel_20EPOCHS_ 2.63LOSS.h5", compile=False)
dModel = load_model(
    "trainedModels/dModel_20EPOCHS_ 2.63LOSS.h5", compile=False)
# print(getResponse("hello how are you"))

loop = asyncio.get_event_loop()


async def start():
  client = EchoBot(loop=loop)
  print("Logging in...")
  await client.start("username", "password")
  print("Listening...")
  client.listen()

loop.run_until_complete(start())
loop.run_forever()
