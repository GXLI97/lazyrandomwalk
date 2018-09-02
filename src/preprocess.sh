CORPUS=text8
DATA_DIR=data
VECTOR_DIR=vectors
GLOVE_BUILD_DIR=GloVe/build

CORPUS_FILE=${DATA_DIR}/${CORPUS}
VOCAB_FILE=${DATA_DIR}/${CORPUS}.vocab.txt
COOCCURRENCE_FILE=${DATA_DIR}/${CORPUS}.cooccurrence.bin
MATRIX_FILE=${DATA_DIR}/${CORPUS}.matrix.npz

WINDOW_SIZE=10
MIN_COUNT=500
MEMORY=16

$GLOVE_BUILD_DIR/vocab_count -verbose 2 -min-count $MIN_COUNT < $CORPUS_FILE > $VOCAB_FILE
$GLOVE_BUILD_DIR/cooccur -verbose 2 -symmetric 1 -window-size $WINDOW_SIZE -vocab-file $VOCAB_FILE -memory $MEMORY -distance-weighting 0 < $CORPUS_FILE > $COOCCURRENCE_FILE
python3 src/matrix.py --vocab_file $VOCAB_FILE --cooccurrence_file $COOCCURRENCE_FILE --matrix_file $MATRIX_FILE

# CORPUS=wikipedia
# DATA_DIR=data
# VECTOR_DIR=vectors
# GLOVE_BUILD_DIR=GloVe/build

# CORPUS_FILE=${DATA_DIR}/${CORPUS}
# VOCAB_FILE=${DATA_DIR}/${CORPUS}.vocab.txt
# COOCCURRENCE_FILE=${DATA_DIR}/${CORPUS}.cooccurrence.bin
# MATRIX_FILE=${DATA_DIR}/${CORPUS}.matrix.npz

# WINDOW_SIZE=10
# MIN_COUNT=1000
# MEMORY=16

# $GLOVE_BUILD_DIR/vocab_count -verbose 2 -min-count $MIN_COUNT < $CORPUS_FILE > $VOCAB_FILE
# $GLOVE_BUILD_DIR/cooccur -verbose 2 -symmetric 1 -window-size $WINDOW_SIZE -vocab-file $VOCAB_FILE -memory $MEMORY -distance-weighting 0 < $CORPUS_FILE > $COOCCURRENCE_FILE
# python3 src/matrix.py --vocab_file $VOCAB_FILE --cooccurrence_file $COOCCURRENCE_FILE --matrix_file $MATRIX_FILE
