import yaml
import os
import pandas as pd
import numpy as np
import math
import random
import re
from sklearn.model_selection import train_test_split
from preprocessing import clean
from preprocessing import taggerTokenizer
from preprocessing import CustomDataGenerator
from wordEmbeddings import generateEmbeddingMatrix


def generateInputTargetData(data, MAX_SENTENCE_LENGTH, path):
  """
  adds <SOS>/<EOS> tokens to targets
  concats sentences longer than MAX_SENTENCE_LENGTH
  creates trainInputs.txt file for training input sentences
  creates trainTarget.txt file for training target sentences
  creates testInputs.txt file for testing input sentences
  creates testTarget.txt file for testing target sentences
  """
  inputs, targets = [], []
  for pair in data:
    cleanText = clean(str(pair[0])).split(" ")
    inputs.append(" ".join(cleanText[:MAX_SENTENCE_LENGTH]))

    cleanText = clean(str(pair[1])).split(" ")
    targets.append(
      "<SOS> " + " ".join(cleanText[:MAX_SENTENCE_LENGTH]) + " <EOS>")

  X_train, X_test, y_train, y_test = train_test_split(
    inputs, targets, test_size=0.05, random_state=1)
  trainI = open(path + "/trainInputs.txt", "w")
  trainT = open(path + "/trainTargets.txt", "w")
  testI = open(path + "/testInputs.txt", "w")
  testT = open(path + "/testTargets.txt", "w")
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


def corpusStats(data):
  """
  Statistics on word count and sentence length of corpus
  """
  dataTemp = [item for sublist in data for item in sublist]
  print("---sentence length statistics---")
  sentenceLength = pd.DataFrame([len(clean(str(line)).split(" "))
                                 for line in dataTemp], columns=["sentenceLength"])
  print(sentenceLength.describe())
  print("\n---word count statistics---")
  wordCounts = dict()
  for line in dataTemp:
    for word in taggerTokenizer(str(line)):
      if word in wordCounts:
        wordCounts[word] += 1
      else:
        wordCounts[word] = 1
  wordCounts = pd.DataFrame.from_dict(wordCounts, orient='index')
  wordCounts.columns = ["wordCounts"]
  wordCounts = wordCounts.sort_values(by=['wordCounts'])
  print(wordCounts.describe())
  return wordCounts, sentenceLength


if __name__ == '__main__':
  np.random.seed(0)
  MAX_VOCAB_SIZE = 1879
  MAX_SENTENCE_LENGTH = 20
  BATCH_SIZE = 25
  EMBEDDING_DIM = 100
  path = "C:/Users/ditta/Desktop/kagglePyBotDataset"
  data = []
  # iterate through all yml files and concat conversations into one list
  for file in os.listdir(path):
    with open(path + "/" + file, 'r') as stream:
      if file.endswith(".yml"):
        try:
          temp = yaml.safe_load(stream)
          data.extend(temp["conversations"])
        except yaml.YAMLError as exc:
          print(exc)
  # print statistics on words in corpus
  corpusStats(data)
  # generate input/target text files and clean data
  generateInputTargetData(data, MAX_SENTENCE_LENGTH, path)

  # Generate embbeding matrix and word2idx dict
  trainGenerator = CustomDataGenerator(
    path + "/trainInputs.txt", path + "/trainTargets.txt", BATCH_SIZE,
    MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE)

  trainGenerator.saveWord2Idx(path + "/word2idx_2.pkl")
  embeddingMatrix = generateEmbeddingMatrix(
    "C:/Users/ditta/Desktop/LSTMPyBotData/glove.6B.100d.txt",
    trainGenerator.word2idx, MAX_SENTENCE_LENGTH, MAX_VOCAB_SIZE, EMBEDDING_DIM,
    path + "/embeddingMatrixGolve6b100_2.npy")
