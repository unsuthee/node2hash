#!/bin/sh

for ds in ng20
do
    for T in 1 5 10 15 20
    do    
        for BITS in 8 16 32 64 128
        do
            python train_EdgeReg_NN.py -g 1 -d $ds -b $BITS -w BFS-20 --num_epochs 50 -T $T --edge_weight 10
        done
    done
done

