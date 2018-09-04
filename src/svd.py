import scipy
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import svds
import time
import pickle
import logging
import argparse
from sklearn.utils.extmath import randomized_svd

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--matrix_file', type=str)
parser.add_argument('--vectors_file', type=str)
parser.add_argument('--dim', type=int)
parser.add_argument('--alpha', type=float)
args = parser.parse_args()

VOCAB_FILE = args.vocab_file
MATRIX_FILE = args.matrix_file
VECTORS_FILE = args.vectors_file
DIM = args.dim
ALPHA = args.alpha

logging.basicConfig(level=logging.INFO)


def truncate(X, xmax):
    '''if values in X are above xmax, set them to xmax.'''
    logging.info("Truncation matrix values above xmax={}".format(xmax))
    X.data[X.data >= xmax] = xmax
    return X


def calculate_pmi(X):
    '''computes the PMI of matrix X'''
    logging.info("Calculating PMI...")
    start = time.time()
    pmi = X
    D = np.array(X.sum(axis=1))[:, 0]
    sum_total = D.sum()
    Dinv = 1/D
    # broadcasting is much faster than matrix mult.
    pmi = np.multiply(Dinv, pmi.T).T
    pmi = np.multiply(pmi, Dinv)
    pmi *= sum_total
    pmi = np.ma.log(pmi)
    pmi = pmi.filled(0)
    end = time.time()
    logging.info("PMI took {} s".format(end-start))
    return pmi


def lazyrandwalk(X):
    ''' compute the lazy random walk operator'''
    if ALPHA == 1.0:
        return X
    D = np.array(X.sum(axis=1))[:,0]
    Dinv = 1/D
    # broadcasting is much much faster than matrix mult.
    R = np.multiply(Dinv, X.T).T
    logging.info("Computing LRW matrix with alpha={}".format(ALPHA))
    start = time.time()
    R = ALPHA * R + (1-ALPHA) * (R ** 2)
    end = time.time()
    logging.info("Time taken {}".format(end-start))
    # broadcasting is much much faster than matrix mult.
    # convert back to p(wi, wj) matrix.
    return np.multiply(D, R.T).T


def write_to_glove_format(u):
    with open(VOCAB_FILE, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(VECTORS_FILE, "w") as f:
        for i in range(u.shape[0]):
            word = words[i]
            f.write(word + " " + " ".join(["%.8f" % (x)
                                           for x in u[i, :]]) + "\n")


def main():
    logging.info("loading matrix from {}".format(MATRIX_FILE))
    X = load_npz(MATRIX_FILE)
    logging.info("First 5 submatrix...")
    logging.info(X[:5,:5])
    logging.info("Dim of matrix X: {}".format(X.shape))
    logging.info("Number of nnz in original matrix: {}".format(X.nnz))
    # convert to probability matrix
    # TODO: below was commented out for numerical stability purposes.
    X.data = X.data/X.sum() 
    # TODO: play with this.
    X = truncate(X, xmax=100)
    logging.info(X[:5,:5])
    X = X.todense()
    
    
    X = lazyrandwalk(X)
    logging.info("Number of non zero after LRW {}".format(np.count_nonzero(X)))
    XPMI = calculate_pmi(X)
    # Compute unweighted SVD on PMI matrix.
    start = time.time()
    logging.info("Calculating SVD...")
    u, s, v = randomized_svd(XPMI, n_components=DIM)
    end = time.time()
    logging.info("SVD took {} s".format(end-start))

    # Save the embeddings.
    write_to_glove_format(u)


if __name__ == "__main__":
    main()
