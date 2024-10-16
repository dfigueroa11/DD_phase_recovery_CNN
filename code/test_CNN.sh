#!/bin/bash

for j in {0..4}; do
    echo "start iterations for the $j loss function"
    for i in {1..2}; do
        echo "Running iteration $i"
        nice -n 5 python3.11 cnn_eq_training_sym.py -m ASK -o 2 -l $j
        nice -n 5 python3.11 cnn_eq_training_sym.py -m ASK -o 4 -l $j
        nice -n 5 python3.11 cnn_eq_training_sym.py -m PAM -o 2 -l $j
        nice -n 5 python3.11 cnn_eq_training_sym.py -m PAM -o 4 -l $j
        nice -n 5 python3.11 cnn_eq_training_sym.py -m QAM -o 4 -l $j
    done
    echo "All iterations for the $j loss function completed."
done
echo "ALL DONE"