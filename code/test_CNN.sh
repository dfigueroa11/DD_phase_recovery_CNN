#!/bin/bash

for i in {0,1,2}
do
    python3.11 fcn_eq_training.py -m ASK -o 2 -l $i &
    python3.11 fcn_eq_training.py -m ASK -o 4 -l $i &
    python3.11 fcn_eq_training.py -m QAM -o 4 -l $i &
    wait
done
echo "ALL DONE"