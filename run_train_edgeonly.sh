#!/bin/sh

for ds in ng20 #agnews citeseer dblp
do
    for T in 1
    do    
        for BITS in 8 16 32 64 128
        do
            if [ "$ds" = "ng20" ]
            then
                python train_EdgeOnly_NN.py -g 2 -d $ds -b $BITS -w BFS-20 --num_epochs 50 -T $T
            elif [ "$ds" = "agnews" ]
            then
                python train_EdgeOnly_NN.py -g 2 -d $ds -b $BITS -w BFS-20 --num_epochs 25 -T $T
            elif [ "$ds" = "dblp" ]
            then
                python train_EdgeOnly.py -g 2 -d $ds -b $BITS -w BFS-20 --num_epochs 3 -T $T
            else
                python train_EdgeOnly.py -g 2 -d $ds -b $BITS -w BFS-20 --num_epochs 50 -T $T
            fi
        done
    done
done

