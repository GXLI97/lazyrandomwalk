DATASET=text8
BUILDDIR=GloVe/build
COOCCURRENCE_FILE=data/$DATASET.cooccur.trunc.bin
COOCCURRENCE_SHUF_FILE=data/$DATASET.cooccur.trunc.shuf.bin
COOCCURRENCE_RW_FILE=data/$DATASET.cooccur.trunc.rw.bin
COOCCURRENCE_SHUF_RW_FILE=data/$DATASET.cooccur.trunc.rw.shuf.bin
VOCAB_FILE=data/$DATASET.vocab.trunc.txt
SAVE_FILE=vectors/$DATASET.glove.vectors
NUM_THREADS=8
MAX_ITER=15
VECTOR_SIZE=50
BINARY=2

#python3 src/write_cooccurrences.py --dataset $DATASET
$BUILDDIR/shuffle -verbose 2 -memory 16.0 < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose 2
python GloVe/eval/python/evaluate.py --vocab_file $VOCAB_FILE --vectors_file $SAVE_FILE.txt

$BUILDDIR/shuffle -verbose 2 -memory 16.0 < $COOCCURRENCE_RW_FILE > $COOCCURRENCE_SHUF_RW_FILE
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_RW_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose 2
python GloVe/eval/python/evaluate.py --vocab_file $VOCAB_FILE --vectors_file $SAVE_FILE.txt