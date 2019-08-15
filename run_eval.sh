#!/bin/sh

for ds in pubmed
do
    for T in 1
    do    
        for BITS in 8 16 32 64 128
        do
            python evaluate_embeddings.py -g 1 -d $ds -b $BITS -T $T --hash
        done
    done
done