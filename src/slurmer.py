split = 10
for i in range(split):
    for j in range(split-i+1):
        weights = [i/split, j/split, (split-i-j)/split]
        name = ','.join(map(str, weights))
        text = '''#!/bin/bash
#SBATCH -N 1  # 1 node
#SBATCH -n 1  # 1 task per node
#SBATCH -c 4
#SBATCH -t 10:00:00 # time required, here it is 1 min
#SBATCH --mem=200G
#SBATCH -o run{}.out # stdout is redirected to that file
#SBATCH -e run{}.err # stderr directed to that file.

DATASET=wikipedia
DIM=500
NUM_WORDS=50000
W1={}
W2={}
W3={}

module load anaconda3
python3 src/svd.py --dataset $DATASET --dim $DIM --num_words $NUM_WORDS --weights $W1 $W2 $W3
python2 GloVe/eval/python/evaluate.py --vocab_file data/${{DATASET}}.vocab.txt --vectors_file vectors/${{DATASET}}{}.vectors.txt
'''.format(name, name, weights[0], weights[1], weights[2], name)
        with open('slurms/batch{}.sh'.format(name), 'w') as f:
            f.write(text)
