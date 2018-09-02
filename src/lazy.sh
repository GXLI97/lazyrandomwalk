CORPUS=text8
DATA_DIR=data
VECTOR_DIR=vectors
GLOVE_BUILD_DIR=GloVe/build

VOCAB_FILE=${DATA_DIR}/${CORPUS}.vocab.txt
MATRIX_FILE=${DATA_DIR}/${CORPUS}.matrix.npz
VECTORS_FILE=${VECTOR_DIR}/${CORPUS}.vectors.txt

DIM=50
NUM_WORDS=1000
ALPHA=0.8

python3 src/svd.py \
    --vocab_file $VOCAB_FILE \
    --matrix_file $MATRIX_FILE \
    --vectors_file $VECTORS_FILE \
    --dim $DIM \
    --num_words $NUM_WORDS \
    --alpha $ALPHA


cd GloVe

python3 ./eval/python/evaluate.py \
    --vocab_file ../$VOCAB_FILE \
    --vectors_file ../$VECTORS_FILE
