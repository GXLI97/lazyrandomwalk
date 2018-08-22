import scipy
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import svds
import time
import pickle
import logging
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
DATASET = args.dataset

COMAT_FILE = "data/{}.comat.npz".format(DATASET)
ID2WORD_FILE = 'data/{}.id2word.p'.format(DATASET)
VECTEXT_FILE = 'vectors/{}.vectors.txt'.format(DATASET)
NUM_WORDS = 3446
ALPHA = 0.85

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

def randwalk(counts):
    logging.info("Calculating Random walk...")
    uni = np.array(counts.sum(axis=1))[:, 0]
    Dinv = dok_matrix((len(uni), len(uni)))
    Dinv.setdiag(1/uni)
    res = Dinv.dot(counts)
    # make matrix dense.
    res = res.todense()
    logging.info("Squaring matrix...")
    start = time.time()
    res = ALPHA * res + (1-ALPHA) * (res ** 2)
    end = time.time()
    logging.info("Time taken {}".format(end-start))
    # convert back to p(w,w') matrix.
    D = dok_matrix((len(uni), len(uni)))
    D.setdiag(uni)
    return D.dot(res)

def main():
    logging.info("loading matrix...")
    X = load_npz(COMAT_FILE)
    X.data = X.data/X.sum()
    X = X[:NUM_WORDS, :NUM_WORDS]
    logging.info(X.nnz)
    logging.info(X.shape)
    Xpow = randwalk(X)

    ids = X.nonzero()
    x = np.squeeze(np.asarray(X[ids]))
    y = np.squeeze(np.asarray(Xpow[ids]))
    fig=plt.figure(figsize=(14, 12), dpi= 80, facecolor='w', edgecolor='k')
    lim = [1e-9, 1e-1]
    plt.loglog(x, y, 'b.')
    plt.loglog(lim, lim, 'r-')
    plt.xlim(lim)
    plt.ylim(lim)
    plt.savefig("plots/vis_pow.png")

    x2 = Xpow[(X==0).nonzero()]
    fig=plt.figure(figsize=(14, 12), dpi= 80, facecolor='w', edgecolor='k')
    hist = plt.hist(np.squeeze(np.asarray(x2)), bins=np.logspace(np.log10(1e-11),np.log10(1e-7), 400), log=True)
    plt.gca().set_xscale("log")
    plt.savefig("plots/pow_new_vals.png")


if __name__ == "__main__":
    main()
