# CORPUS=text8
# DATA_DIR=data
# VECTOR_DIR=vectors
# GLOVE_BUILD_DIR=GloVe/build

CORPUS=wikipedia
DATA_DIR=/scratch/network/gxli/data
VECTOR_DIR=/scratch/network/gxli/vectors
GLOVE_BUILD_DIR=GloVe/build

VOCAB_FILE=${DATA_DIR}/${CORPUS}.vocab.txt
COOCCURRENCE_FILE=${DATA_DIR}/${CORPUS}.cooccurrence.bin
COOCCURRENCE_SHUF_FILE=${DATA_DIR}/${CORPUS}.cooccurrence.shuf.bin
VECTORS_FILE=${VECTOR_DIR}/${CORPUS}.glove.vectors
TEMP_FILE=${DATA_DIR}/${CORPUS}.temp_shuffle

MEMORY=16.0
DIM=500

$GLOVE_BUILD_DIR/shuffle \
    -verbose 2 \
    -memory $MEMORY \
    -temp-file $TEMP_FILE \
    < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

$GLOVE_BUILD_DIR/glove \
    -verbose 2 \
    -vector-size $DIM \
    -eta 0.01 \
    -input-file $COOCCURRENCE_SHUF_FILE \
    -vocab-file $VOCAB_FILE \
    -save-file $VECTORS_FILE

cd GloVe

python3 ./eval/python/evaluate.py \
    --vocab_file $VOCAB_FILE \
    --vectors_file $VECTORS_FILE.txt
