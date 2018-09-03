CORPUS=text8
DATA_DIR=data
VECTOR_DIR=vectors
GLOVE_BUILD_DIR=GloVe/build

VOCAB_FILE=${DATA_DIR}/${CORPUS}.vocab.txt
COOCCURRENCE_FILE=${DATA_DIR}/${CORPUS}.cooccurrence.bin
COOCCURRENCE_FILE=${DATA_DIR}/${CORPUS}.cooccurrence.bin
VECTORS_FILE=${VECTOR_DIR}/${CORPUS}.glove.vectors.txt
TEMP_FILE=${DATA_DIR}/${CORPUS}.temp_shuffle

MEMORY=128
DIM=500

$GLOVE_BUILD_DIR/shuffle \
    -verbose $VERBOSE \
    -memory $MEMORY \
    -temp-file
    < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

$GLOVE_BUILD_DIR/glove \
    -verbose 2 \
    -vector-size $DIM \
    -input-file $COOCCURRENCE_SHUF_FILE
    -vocab-file $VOCAB_FILE
    -save-file $VECTORS_FILE

cd GloVe

python3 ./eval/python/evaluate.py \
    --vocab_file ../$VOCAB_FILE \
    --vectors_file ../$VECTORS_FILE
