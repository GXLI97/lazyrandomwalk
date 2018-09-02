#!/bin/bash
#SBATCH -N 1  # 1 node
#SBATCH -n 1  # 1 task per node
#SBATCH -c 4
#SBATCH -t 10:00:00 # time required, here it is 1 min
#SBATCH --mem=200G
#SBATCH -o slurms/logs/wiki.out # stdout is redirected to that file
#SBATCH -e slurms/logs/wiki.err # stderr is redirected to that file

cd data
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
bzip2 -d enwiki-latest-pages-articles.xml.bz2
perl ../src/wiki_preprocess.pl enwiki-latest-pages-articles.xml > wikipedia
rm enwiki-latest-pages-articles.xml.bz2
rm enwiki-latest-pages-articles.xml