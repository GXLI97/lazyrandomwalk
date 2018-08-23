## A Lazy Random Walk Operator for Matrix Factorization Word Embeddings

Download GloVe:

```
git clone https://github.com/stanfordnlp/GloVe.git
cd GloVe
make
```

Download text8.
```
mkdir data
cd data
if [ ! -e text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi
```

Download wikipedia.
```
cd data
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
bzip2 -d enwiki-latest-pages-articles.xml.bz2
perl ../src/wiki_preprocess.pl enwiki-latest-pages-articles.xml > wikipedia
rm enwiki-latest-pages-articles.xml.bz2
rm enwiki-latest-pages-articles.xml
```

Run on text8 example.


