CORPUS=data/text8
VOCAB_FILE=data/text8.vocab.txt
COOCCURRENCE_FILE=data/text8.cooccurrence.bin
BUILDDIR=GloVe/build
VERBOSE=2
MIN_COUNT=500

echo "$BUILDDIR/vocab_count -verbose $VERBOSE -min-count $MIN_COUNT < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -verbose $VERBOSE -min-count $MIN_COUNT < $CORPUS > $VOCAB_FILE
echo "$BUILDDIR/cooccur -verbose $VERBOSE -symmetric 1 -window-size 10 -vocab-file $VOCAB_FILE -memory 12 -distance-weighting 0 < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -verbose $VERBOSE -symmetric 1 -window-size 10 -vocab-file $VOCAB_FILE -memory 12 -overflow-file tempoverflow -distance-weighting 0 < $CORPUS > $COOCCURRENCE_FILE
