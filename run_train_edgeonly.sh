#!/bin/sh

for ds in cora
do
    for T in 1
    do    
        for BITS in 8 16 32 64 128
        do
            python train_EdgeOnly.py -g 1 -d $ds -b $BITS -w BFS-20 --num_epochs 50 -T $T
        done
    done
done

