#!/bin/bash
clear

echo "$@"

for r in 1
do
    for t in Random DFS BFS
    do
        for i in 1 5 10 20 50 100
        do
            python train_EdgeReg.py -b 32 -g 1 -d "$@" -e 10 "-w $t-$i" --edge_weight 1.0
        done
    done
done
