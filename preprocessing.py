import pandas as pd
import numpy as np
import math
import random
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.utils import Sequence


def clean(text):

  text = text.lower()
  text = re.sub(r"i'm", "i am", text)
  text = re.sub(r"he's", "he is", text)
  text = re.sub(r"she's", "she is", text)
  text = re.sub(r"it's", "it is", text)
  text = re.sub(r"that's", "that is", text)
  text = re.sub(r"what's", "that is", text)
  text = re.sub(r"where's", "where is", text)
  text = re.sub(r"how's", "how is", text)
  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"won't", "will not", text)
  text = re.sub(r"can't", "cannot", text)
  text = re.sub(r"n't", " not", text)
  text = re.sub(r"n'", "ng", text)
  text = re.sub(r"'bout", "about", text)
  text = re.sub(r"'til", "until", text)
  text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
  text = " ".join(text.split())
  return text


def convPairGenerator(convNum, convs, idx2line):
  """
  Generates input/target pairs given a conversation number and list of convos
  e.g. if conversation=[Line1,Line2,Line3]
  returns [(Line1, Line2), (Line2, Line3)]
  """
  pairs = []
  for i in range(len(convs[convNum])):
    if i < len(convs[convNum]) - 1:
      pairs.append((idx2line[convs[convNum][i]],
                    idx2line[convs[convNum][i + 1]]))
  return pairs


def generateInputTargetData(idx2line, convs, MAX_SENTENCE_LENGTH):
  """
  adds <SOS>/<EOS> tokens to targets
  concats sentences longer than MAX_SENTENCE_LENGTH
  creates trainInputs.txt file for training input sentences
  creates trainTarget.txt file for training target sentences
  creates testInputs.txt file for testing input sentences
  creates testTarget.txt file for testing target sentences
  """
  inputs = []
  targets = []
  for conv in range(len(convs)):
    linePairs = convPairGenerator(conv, convs, idx2line)
    for pair in linePairs:
      cleanText = clean(pair[0]).split(" ")
      inputs.append(" ".join(cleanText[:MAX_SENTENCE_LENGTH]))

      cleanText = clean(pair[1]).split(" ")
      targets.append(
        "<SOS> " + " ".join(cleanText[:MAX_SENTENCE_LENGTH]) + " <EOS>")

  X_train, X_test, y_train, y_test = train_test_split(
    inputs, targets, test_size=0.2, random_state=1)
  trainI = open("data/trainInputs.txt", "w")
  trainT = open("data/trainTargets.txt", "w")
  testI = open("data/testInputs.txt", "w")
  testT = open("data/testTargets.txt", "w")
  for sentence in X_train:
    trainI.write(sentence + "\n")
  for sentence in y_train:
    trainT.write(sentence + "\n")
  for sentence in X_test:
    testI.write(sentence + "\n")
  for sentence in y_test:
    testT.write(sentence + "\n")
  trainI.close()
  trainT.close()
  testI.close()
  testT.close()


def preprocess(linePath, convPath, MAX_SENTENCE_LENGTH):

  print("Starting preprocessing...")
  # read in raw movie lines and conversation lines from Cornel Movie Corpus
  lines = open(linePath, encoding='utf-8',
               errors='ignore').read().split('\n')
  conversations = open(convPath, encoding='utf-8',
                       errors='ignore').read().split('\n')
  # extract line labels and line text
  idx2line = {}
  for line in lines:
    temp = line.split(' +++$+++ ')
    if len(temp) == 5:
      idx2line[temp[0]] = temp[4]
  convs = []
  for line in conversations[:-1]:
    temp = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    convs.append(temp.split(','))

  generateInputTargetData(idx2line, convs, MAX_SENTENCE_LENGTH)
  print("Preprocessing finished\n")
  return idx2line, convs


def taggerTokenizer(text):
  text = "<SOS> " + clean(text) + " <EOS>"
  return text.split(" ")


def corpusStats(idx2line):
  """
  Statistics on word count and sentence length of corpus
  """
  print("---sentence length statistics---")
  sentenceLength = pd.DataFrame([len(clean(line).split(" "))
                                 for line in idx2line.values()], columns=["sentenceLength"])
  print(sentenceLength.describe())
  print("\n---word count statistics---")
  wordCounts = dict()
  for line in idx2line.values():
    for word in taggerTokenizer(line):
      if word in wordCounts:
        wordCounts[word] += 1
      else:
        wordCounts[word] = 1
  wordCounts = pd.DataFrame.from_dict(wordCounts, orient='index')
  wordCounts.columns = ["wordCounts"]
  wordCounts = wordCounts.sort_values(by=['wordCounts'])
  print(wordCounts.describe())
  # wordCounts = wordCounts[wordCounts["counts"] >= int(wordCounts.mean())]
  # wordCounts = wordCounts[wordCounts["counts"] >= 15]
  # vocab = list(wordCounts.index)


class CornellMovieDataGenerator(Sequence):
  """
  Data generator for Cornell Movie-Dialogs Corpus
  reads in data in batches so that using to_categorical
  on decoderTarget doesnt cause emeory issues
  """

  def __init__(self, inputPath, targetPath, batchSize,
               MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE, shuffle=True):

    self.inputPath = inputPath
    self.targetPath = targetPath
    self.batchSize = batchSize
    self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
    self.MAX_VOCAB_SIZE = MAX_VOCAB_SIZE
    self.shuffle = shuffle

    with open(inputPath, "r") as f:
      self.numLines = sum(1 for line in f)
    # since linecache starts index at 1
    self.lineIDs = list(range(1, self.numLines + 1))
    self.tokenizer, self.word2idx = self.initTokenizer()

  def __len__(self):
    return int(np.ceil(self.numLines / float(self.batchSize)))

  def __getitem__(self, idx):
    """
    generates 1 batch given a batch number
    """
    batchLines = self.lineIDs[idx * self.batchSize:(idx + 1) * self.batchSize]

    batchInputs, batchTargets = [], []
    for line in batchLines:
      batchInputs.append(linecache.getline(self.inputPath, line))
      batchTargets.append(linecache.getline(self.targetPath, line))

    encoderInput, decoderInput, decoderTargets = self.tokenizeData(
      batchInputs, batchTargets)

    return [encoderInput, decoderInput], decoderTargets

  def on_epoch_end(self):
    if self.shuffle == True:
      np.random.shuffle(self.lineIDs)

  def initTokenizer(self):
    print("initializing tokenizer...")
    inputs = open(self.inputPath, encoding="utf-8",
                  errors="ignore").read().split("\n")
    targets = open(self.targetPath, encoding="utf-8",
                   errors="ignore").read().split("\n")
    # remove empty last lines
    inputs = inputs[:-1]
    targets = targets[:-1]

    # update tokenizer with cornell movie text corpus
    tokenizer = Tokenizer(num_words=self.MAX_VOCAB_SIZE, filters="",
                          lower=False, oov_token="<UNK>")
    tokenizer.fit_on_texts(inputs + targets)

    # create dict mapping words to integers
    word2idx = tokenizer.word_index
    return tokenizer, word2idx

  def tokenizeData(self, batchInputs, batchTargets):
    """
    encoderInput: "hello how are you"->[12,32,34,5,0,0,0]
      sentences that feed into the encoder

    decoderInput: "<SOS> I am good <EOS>"->[1,42,13,51,2,0,0,0]
      sentences that feed into the decoder

    decoderTarget: "I am good <EOS>"->[42,13,51,2,0,0,0,0]
    decoderTarget1Hot: one hot vectorizes decoderTarget
      targets that the decoder tries to learn to ouput
    """
    batchInputs = [line.strip("\n") for line in batchInputs]
    batchTargets = [line.strip("\n") for line in batchTargets]
    # convert text to sequences of integers
    # for decoderTargets remove <SOS> for each sentence
    decoderTargets = self.tokenizer.texts_to_sequences(
      [x.split(' ', 1)[1] for x in batchTargets])
    decoderInput = self.tokenizer.texts_to_sequences(batchTargets)
    encoderInput = self.tokenizer.texts_to_sequences(batchInputs)

    # pad integer sequences
    decoderTargets = pad_sequences(
        decoderTargets, maxlen=self.MAX_SENTENCE_LENGTH, padding="post")
    decoderInput = pad_sequences(
        decoderInput, maxlen=self.MAX_SENTENCE_LENGTH, padding="post")
    encoderInput = pad_sequences(
        encoderInput, maxlen=self.MAX_SENTENCE_LENGTH, padding="post")

    # create one hot encodings of decoderTargets
    decoderTargets = np.array(to_categorical(
      decoderTargets, self.MAX_VOCAB_SIZE))

    return encoderInput, decoderInput, decoderTargets


if __name__ == '__main__':
  idx2line, convs = preprocess(
    "data/movie_lines.txt", "data/movie_conversations.txt", 20)
  corpusStats(idx2line)
