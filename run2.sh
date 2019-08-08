#!/bin/sh

for ds in dblp
do
    for T in 1 5 10 15 20
    do    
        for BITS in 8 16 32 64 128
        do
            python train_EdgeReg.py -g 1 -d $ds -b $BITS -w BFS-20 --num_epochs 3 -T $T
        done
    done
done

