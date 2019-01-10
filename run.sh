#!/bin/sh
for BITS in 8 16 32 64 128
do
    python train_EdgeReg.py -g 3 -d ng20 -b $BITS -w BFS-50 --topn 20 --num_epochs 25
done

