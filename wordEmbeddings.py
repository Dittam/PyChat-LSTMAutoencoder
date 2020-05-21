import numpy as np


def generateUNKEmbedding(embeddingVecs, EMBEDDING_DIM):
  """
  Given a list of embedding vectors
  Generates embedding vector for the unkown token <UNK> by taking the
  average of all embedding vectors provided by Glove
  """
  allEmbeddings = np.zeros(
    (len(embeddingVecs), EMBEDDING_DIM), dtype=np.float32)
  for i, vec in enumerate(embeddingVecs):
    allEmbeddings[i] = vec
  averageVec = np.mean(allEmbeddings, axis=0)
  return averageVec, allEmbeddings


def generateEmbeddingMatrix(gloveEmbeddingPath, word2idx, MAX_SENTENCE_LENGTH,
                            MAX_VOCAB_SIZE, EMBEDDING_DIM, saveMatrix=False):
  """
  Generates matrix where columns are embedding vectors for each word in
  corpus vocabulary
  word2idx is a dict mapping words to integers, generated from
  preprocessing.tokenizeData()
  """
  print("Generating word embedding matrix...")
  # Import Glove embedding vectors and create a dict mapping words to embeding
  # vectors
  word2embedding = {}
  with open(gloveEmbeddingPath, encoding="utf8") as f:
    for line in f:
      word, coefs = line.split(maxsplit=1)
      coefs = np.fromstring(coefs, "f", sep=" ")
      word2embedding[word] = coefs

  # initialize embedding matrix with 0
  embeddingMatrix = np.zeros((MAX_VOCAB_SIZE, EMBEDDING_DIM))
  unkEmbeddingVec, allEmbeddings = generateUNKEmbedding(
    list(word2embedding.values()), EMBEDDING_DIM)
  for word, idx in word2idx.items():
    if idx >= MAX_VOCAB_SIZE:
      break
    vec = word2embedding.get(word)
    # generate random embedding for these tokens
    if word == "<SOS>" or word == "<EOS>":
      embeddingMatrix[idx] = np.random.uniform(low=np.min(
        allEmbeddings), high=np.max(allEmbeddings), size=(EMBEDDING_DIM,))
    elif vec is None:
      embeddingMatrix[idx] = unkEmbeddingVec
    else:
      embeddingMatrix[idx] = vec

  if saveMatrix != "":
    np.save(saveMatrix, embeddingMatrix)

  return embeddingMatrix


if __name__ == '__main__':
  np.random.seed(0)
  import preprocessing
  trainGenerator = preprocessing.CornellMovieDataGenerator(
    "data/trainInputs.txt", "data/trainTargets.txt", 1024, 20, 10000)
  embeddingMatrix = generateEmbeddingMatrix(
    "data/glove.6B.100d.txt", trainGenerator.word2idx, 20, 10000, 100, 
    "embeddingMatrixGolve6b100_2.npy")
