#!/bin/bash

for ((i=0; i < $1; i++)); do
    awk '$3 == 0{print $1,$2}' spike_result_trial$i.dat > ../../../EvalPOT/gr_spike/gr_$i.dat
done;
