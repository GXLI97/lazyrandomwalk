import scipy
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import svds
import time
import pickle
import logging
import argparse
import struct

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
DATASET = args.dataset
COMAT_FILE = "data/{}.comat.npz".format(DATASET)
VOCAB_TXT_FILE = "data/{}.vocab.txt".format(DATASET)
VOCAB_TXT_FILE_TRUNC = "data/{}.vocab.trunc.txt".format(DATASET)
COOCCUR_TRUNC_BIN_FILE = "data/{}.cooccur.trunc.bin".format(DATASET)
COOCCUR_TRUNC_RW_BIN_FILE = "data/{}.cooccur.trunc.rw.bin".format(DATASET)
NUM_WORDS = 3446
ALPHA=0.75

logging.info(ALPHA)

# first we read vocab and write to a text file of the truncated vocab...
with open(VOCAB_TXT_FILE, 'r') as f:
    words = f.readlines()
    words = words[:NUM_WORDS]

with open(VOCAB_TXT_FILE_TRUNC, 'w') as f:
    for word in words:
        f.write(word)

# load the matrix into python.
logging.info("loading matrix...")
X = load_npz(COMAT_FILE)
# convert to probability matrix
X.data = X.data/X.sum()
X = X[:NUM_WORDS, :NUM_WORDS]
X = X.todense()
logging.info(X.shape)

logging.info("writing to bin...")

start = time.time()
with open(COOCCUR_TRUNC_BIN_FILE, 'wb') as f:
    for i in range(NUM_WORDS):
        for j in range(NUM_WORDS):
            if X[i,j] != 0:
                f.write(struct.pack("i", i+1))
                f.write(struct.pack("i", j+1))
                f.write(struct.pack("d", X[i,j]))      
end = time.time()
logging.info(end-start)

def validate():
    # Validate on text8.
    with open(COOCCUR_TRUNC_BIN_FILE, "rb") as f:
        i = 0
        while True:
            try:
                id1 = struct.unpack('i', f.read(4))[0]
                id2 = struct.unpack('i', f.read(4))[0]
                c = struct.unpack('d', f.read(8))[0]
                i += 1
            except:
                break
            if X[id1-1, id2-1] != c:
                print(id1, id2, X[id1-1, id2-1], c)
            if i == 100:
                break
    logging.info("DONE VALIDATING")

validate()

def randwalk(counts):
    logging.info("Calculating Random walk...")
    uni = np.array(counts.sum(axis=1))[:, 0]
    Dinv = dok_matrix((len(uni), len(uni)))
    Dinv.setdiag(1/uni)
    res = Dinv.dot(counts)
    # make matrix dense.
    # res = res.todense()
    logging.info("Squaring matrix...")
    start = time.time()
    res = ALPHA * res + (1-ALPHA) * (res ** 2)
    end = time.time()
    logging.info("Time taken {}".format(end-start))
    # convert back to p(w,w') matrix.
    D = dok_matrix((len(uni), len(uni)))
    D.setdiag(uni)
    return D.dot(res)

Xrw = randwalk(X)

start = time.time()
with open(COOCCUR_TRUNC_RW_BIN_FILE, 'wb') as f:
    for i in range(NUM_WORDS):
        for j in range(NUM_WORDS):
            f.write(struct.pack("i", i+1))
            f.write(struct.pack("i", j+1))
            f.write(struct.pack("d", Xrw[i,j]))
end = time.time()
logging.info(end-start)
