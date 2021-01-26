#!/bin/bash

CPU_SIM=0
TAU=0
i=0
#for CPU_SIM in $(seq 10 10 50)
#do
#    for i in $(seq 1 4)
#    do
#        ./main 2000 $CPU_SIM $((CPU_SIM * 2)) $((CPU_SIM * i / 2))
#    done
#done


for TAU in $(seq 20 20 100)
do
    for CPU_SIM in $(seq 10 10 50)
    do
        ./main 2000 $CPU_SIM $((CPU_SIM * 2)) $TAU
    done
done
