#!/bin/sh

for ds in reddit
do
    for T in 1 5 10 15 20
    do    
        for BITS in 8 16 32 64 128
        do
            python train_EdgeReg.py -g 3 -d $ds -b $BITS -w BFS-20 --num_epochs 3 -T $T --edge_weight 100
        done
    done
done

