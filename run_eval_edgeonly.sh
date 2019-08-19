#!/bin/sh

for ds in cora citeseer
do
    for T in 1
    do    
        for BITS in 8 16 32 64 128
        do
            if [ "$ds" = "ng20" ]
            then
                python evaluate_embeddings_NN_edgeonly.py -g 0 -d $ds -b $BITS -T $T --hash
            elif [ "$ds" = "agnews" ]
            then
                python evaluate_embeddings_NN_edgeonly.py -g 0 -d $ds -b $BITS -T $T --hash
            elif [ "$ds" = "dblp" ]
            then
                python evaluate_embeddings_edgeonly.py -g 0 -d $ds -b $BITS -T $T --batch_size 25 --hash
            else
                python evaluate_embeddings_edgeonly.py -g 0 -d $ds -b $BITS -T $T --batch_size 100 --hash
            fi
        done
    done
done
