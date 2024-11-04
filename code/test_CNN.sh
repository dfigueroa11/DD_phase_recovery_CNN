#!/bin/bash

# for j in {0..4}; do
#     echo "start iterations for the $j loss function"
    for i in {1..1}; do
        echo "Running iteration $i"
        nice -n 5 python3.11 cnn_eq_training_complexity.py -m ASK -o 2 -l 5
        nice -n 5 python3.11 cnn_eq_training_complexity.py -m ASK -o 4 -l 5
        nice -n 5 python3.11 cnn_eq_training_complexity.py -m PAM -o 2 -l 1
        nice -n 5 python3.11 cnn_eq_training_complexity.py -m PAM -o 4 -l 1
        nice -n 5 python3.11 cnn_eq_training_complexity.py -m QAM -o 4 -l 0
    done
#     echo "All iterations for the $j loss function completed."
# done
echo "ALL DONE"