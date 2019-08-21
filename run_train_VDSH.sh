#!/bin/sh

for ds in ng20 dblp
do
    for T in 1
    do    
        for BITS in 8 16 32 64 128
        do
            if [ "$ds" = "ng20" ]
            then
                python train_VDSH_NN.py -g 3 -d $ds -b $BITS --num_epochs 50 -T $T
            elif [ "$ds" = "agnews" ]
            then
                python train_VDSH_NN.py -g 3 -d $ds -b $BITS --num_epochs 25 -T $T
            elif [ "$ds" = "dblp" ]
            then
                python train_VDSH.py -g 3 -d $ds -b $BITS --num_epochs 3 -T $T
            else
                python train_VDSH.py -g 3 -d $ds -b $BITS --num_epochs 50 -T $T
            fi
        done
    done
done

