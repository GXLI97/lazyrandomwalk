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
parser.add_argument('--num_words', type=int)
parser.add_argument('--alpha', type=float)
args = parser.parse_args()

VOCAB_FILE = args.vocab_file
MATRIX_FILE = args.matrix_file
VECTORS_FILE = args.vectors_file
DIM = args.dim
NUM_WORDS = args.num_words
ALPHA = args.alpha

logging.basicConfig(level=logging.INFO)


def truncate(X, xmax):
    '''if values in X are above xmax, set them to xmax.'''
    X.data[X.data > xmax] = xmax
    return X


def calculate_pmi(X):
    '''computes the PMI of matrix X'''
    logging.info("Calculating PMI...")
    # pmi = csr_matrix(X)
    start = time.time()
    pmi = X
    sum_w = np.array(X.sum(axis=1))[:, 0]
    # sum_c = np.array(X.sum(axis=0))[0, :]
    sum_total = sum_w.sum()
    sum_w = 1/sum_w
    # print(sum_w)
    sum_w = np.diag(sum_w)
    # sum_c = 1/sum_c

    # normL = dok_matrix((len(sum_w), len(sum_w)))
    # normL.setdiag(sum_w)
    # normR = dok_matrix((len(sum_c), len(sum_c)))
    # normR.setdiag(sum_c)

    # norm = dok_matrix((len(sum_w), len(sum_w)))
    # norm.setdiag(sum_w)

    pmi = np.ma.log(sum_w * pmi * sum_w * sum_total)
    pmi = pmi.filled(0)
    # pmi = pmi * sum_w
    # print(pmi.shape)
    end = time.time()
    logging.info("PMI took {} s".format(end-start))
    # pmi = norm.tocsr().dot(pmi)
    # print('two')
    # pmi = pmi.dot(norm.tocsr())
    # print('three')
    # pmi *= sum_total
    # print("four")
    # pmi = normL.tocsr().dot(pmi).dot(normR.tocsr()) * sum_total
    # pmi.data = np.log(pmi.data)
    return pmi


def lazyrandwalk(X):
    ''' compute the lazy random walk operator'''
    if ALPHA == 1.0:
        return X
    uni = np.array(X.sum(axis=1))[:, 0]
    
    # Dinv = dok_matrix((len(uni), len(uni)))
    # Dinv.setdiag(1/uni)
    Dinv = np.diag(1/uni)
    # R = np.multiply(Dinv[:,None], X)
    R = Dinv * X
    # make matrix dense.
    # R = R.todense()
    logging.info("Computing LRW matrix with alpha={}".format(ALPHA))
    start = time.time()
    # TODO: figure out numwords issue.
    # TODO: try the approximation to reduce size of matrix.
    R = ALPHA * R + (1-ALPHA) * (R ** 2)
    # convert back to p(w,w') matrix.
    # D = dok_matrix((len(uni), len(uni)))
    # D.setdiag(uni)
    D = np.diag(uni)
    end = time.time()
    logging.info("Time taken {}".format(end-start))
    # return np.multiply(D[:,None], R)
    return D*R
    # return D.dot(R)


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
    # convert to probability matrix
    X.data = X.data/X.sum()

    X = X.todense()

    # TODO: figure out the numwords issue.
    # X = X[:NUM_WORDS, :NUM_WORDS]
    logging.info("Dim of matrix X: {}".format(X.shape))
    # Apply preprocessing steps to X.
    # X = truncate(X)
    X = lazyrandwalk(X)
    XPMI = calculate_pmi(X)
    # Compute unweighted SVD on PMI matrix.
    start = time.time()
    logging.info("Calculating SVD...")
    # u, s, vt = svds(csc_matrix(X), k=DIM)
    u, s, v = randomized_svd(XPMI, n_components=DIM)
    end = time.time()
    logging.info("SVD took {} s".format(end-start))

    # Save the embeddings.
    write_to_glove_format(u)


if __name__ == "__main__":
    main()
