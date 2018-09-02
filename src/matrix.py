import struct
from scipy.sparse import dok_matrix, csr_matrix, save_npz
import numpy as np
import pickle
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logging.info("WRITING TO NPZ MATRIX")

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_file", type=str)
parser.add_argument("--cooccurrence_file", type=str)
parser.add_argument("--matrix_file", type=str)
args = parser.parse_args()

VOCAB_FILE = args.vocab_file
COOCCURRENCE_FILE = args.cooccurrence_file
MATRIX_FILE = args.matrix_file

logging.info("Started reading vocab file {}".format(VOCAB_FILE))
with open(VOCAB_FILE) as f:
    words = f.readlines()
n = len(words)
logging.info("Number of words in vocabulary: {}".format(n))



mat = csr_matrix((n, n), dtype=np.int32)
tmp_mat = dok_matrix((n, n), dtype=np.int32)
update_threshold = 1000000

logging.info("Building cooccurrence matrix from {}".format(COOCCURRENCE_FILE))
i = 0
with open(COOCCURRENCE_FILE, "rb") as f:
    while True:
        try:
            # subtract one because GloVe indexing starts at 1.
            id1 = struct.unpack('i', f.read(4))[0] - 1
            id2 = struct.unpack('i', f.read(4))[0] - 1
            c = struct.unpack('d', f.read(8))[0]
        except:
            break
        tmp_mat[id1, id2] = int(c)
        i += 1
        if i % update_threshold == 0:
            print("Processed {} pairs".format(i), end="\r")
            mat = mat + tmp_mat.tocsr()
            tmp_mat = dok_matrix((n,n), dtype=np.int32)
    mat = mat + tmp_mat.tocsr()
logging.info("Saving matrix to {}".format(MATRIX_FILE))
save_npz(MATRIX_FILE, mat)
