CORPUS=data/text8
VOCAB_FILE=data/text8.vocab.txt
COOCCURRENCE_FILE=data/text8.cooccurrence.bin
BUILDDIR=GloVe/build
MIN_COUNT=500

$BUILDDIR/vocab_count -verbose 2 -min-count $MIN_COUNT < $CORPUS > $VOCAB_FILE
$BUILDDIR/cooccur -verbose 2 -symmetric 1 -window-size 10 -vocab-file $VOCAB_FILE -memory 12 -overflow-file tempoverflow -distance-weighting 0 < $CORPUS > $COOCCURRENCE_FILE
