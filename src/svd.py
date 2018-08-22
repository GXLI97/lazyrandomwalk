import scipy
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import svds
import time
import pickle
import logging
import argparse

# python3 src/svd.py --dataset text8 --dim 50 --num_words 3446 --weights 0.7 0.2 0.1
# python3 src/svd.py --dataset wikipedia --dim 500 --num_words 10000 --weights 0.7 0.2 0.1
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--dim', type=int)
parser.add_argument('--num_words', type=int)
parser.add_argument('--weights', type=float, nargs="+")
args = parser.parse_args()


DATASET = args.dataset
COMAT_FILE = "data/{}.comat.npz".format(DATASET)
ID2WORD_FILE = 'data/{}.id2word.p'.format(DATASET)

DIM = args.dim
NUM_WORDS = args.num_words
# XMAX = 1e10

WEIGHTS = args.weights
name = ','.join(map(str, WEIGHTS))
print(WEIGHTS)
VECTEXT_FILE = 'vectors/{}{}.vectors.txt'.format(DATASET, name)

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

# def truncate(counts):
#     counts.data[counts.data > XMAX] = XMAX
#     return counts

def calc_pmi(counts):
    logging.info("Calculating PMI...")
    pmi = csr_matrix(counts)
    logging.info(pmi.shape)
    logging.info(pmi.getnnz())
    
    sum_w = np.array(counts.sum(axis=1))[:, 0]
    sum_c = np.array(counts.sum(axis=0))[0, :]
    sum_total = sum_c.sum()
    sum_w = 1/sum_w
    sum_c = 1/sum_c

    normL = dok_matrix((len(sum_w), len(sum_w)))
    normL.setdiag(sum_w)
    normR = dok_matrix((len(sum_c), len(sum_c)))
    normR.setdiag(sum_c)
    
    pmi = normL.tocsr().dot(pmi).dot(normR.tocsr()) * sum_total
    pmi.data = np.log(pmi.data)
    return pmi

def randwalk(counts):
    logging.info("Calculating Random walk...")
    uni = np.array(counts.sum(axis=1))[:, 0]
    Dinv = dok_matrix((len(uni), len(uni)))
    Dinv.setdiag(1/uni)
    res = Dinv.dot(counts)
    # make matrix dense.
    res = res.todense()
    logging.info("Powering matrix...")
    start = time.time()
    res = WEIGHTS[0] * res + WEIGHTS[1] * (res ** 2) + WEIGHTS[2] * (res**3)
    end = time.time()
    logging.info("Time taken {}".format(end-start))
    # convert back to p(w,w') matrix.
    D = dok_matrix((len(uni), len(uni)))
    D.setdiag(uni)
    return D.dot(res)

def squaring_fun(counts):
    counts = counts.todense()
    beta = 3446
    logging.info("Lets have fun squares")
    logging.info(counts[-beta:,-beta:].shape)
    counts[-beta:,-beta:] = counts[-beta:,-beta:] ** 2
    logging.info("this squaring is done...")
    return counts

#-----
# write to file so we can use glove eval code.
def write_to_glove_format(u):
    id2word = pickle.load(open(ID2WORD_FILE, "rb"))
    with open(VECTEXT_FILE, "w") as f:
        for i in range(u.shape[0]):
            word = id2word[i]
            f.write(word + " " + " ".join(["%.6f" % (x) for x in u[i,:]]) + "\n")

def main():
    logging.info("loading matrix...")
    X = load_npz(COMAT_FILE)
    # convert to probability matrix
    X.data = X.data/X.sum()
    X = X[:NUM_WORDS, :NUM_WORDS]
    logging.info(X.shape)
    # Apply preprocessing steps to X.
    # X = truncate(X)
    X = randwalk(X)
    # X = squaring_fun(X)
    XPMI = calc_pmi(X)

    # Compute unweighted SVD on PMI matrix.
    start = time.time()
    logging.info("Calculating SVD...")
    u, s, vt = svds(XPMI.tocsc(), k=DIM)
    end = time.time()
    logging.info("SVD took {} s".format(end-start))

    # Save the embeddings.
    np.save("vectors/{}.u.npy".format(DATASET), u)
    np.save("vectors/{}.s.npy".format(DATASET), s)
    np.save("vectors/{}.vt.npy".format(DATASET), vt)
    write_to_glove_format(u)

if __name__ == "__main__":
    main()

