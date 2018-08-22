import struct
from scipy.sparse import dok_matrix, csr_matrix, save_npz
import numpy as np
import pickle

DATA = "text8"

VOCAB_FILE = 'data/{}.vocab.txt'.format(DATA)
COOCCUR_FILE = 'data/{}.cooccurrence.bin'.format(DATA)
WORD2ID_FILE = 'data/{}.word2id.p'.format(DATA)
ID2WORD_FILE = 'data/{}.id2word.p'.format(DATA)
COMAT_FILE = 'data/{}.comat.npz'.format(DATA)

with open(VOCAB_FILE) as f:
    words = f.readlines()

n = len(words)

word2id = {}
id2word = []

id = 0
for entry in words:
    w, c = entry.strip().split()
    word2id[w] = id
    id2word.append(w)
    id += 1

pickle.dump(word2id, open(WORD2ID_FILE, "wb"))
pickle.dump(id2word, open(ID2WORD_FILE, "wb"))

counts = csr_matrix((n, n), dtype=np.int32)
tmp_counts = dok_matrix((n, n), dtype=np.int32)
update_threshold = 1000000

i = 0
with open(COOCCUR_FILE, "rb") as f:
    while True:
        try:
            id1 = struct.unpack('i', f.read(4))[0] - 1
            id2 = struct.unpack('i', f.read(4))[0] - 1
            c = struct.unpack('d', f.read(8))[0]
        except:
            break
        tmp_counts[id1, id2] = int(c)
        i += 1
        if i % update_threshold == 0:
            print("processed {} pairs".format(i), end="\r")
            counts = counts + tmp_counts.tocsr()
            tmp_counts = dok_matrix((n,n), dtype=np.int32)
        
    counts = counts + tmp_counts.tocsr()
    
save_npz(COMAT_FILE, counts)