#!/bin/bash

for i in {1..5}; do
echo "Running iteration $i"
#####################################
nice -n 5 python3.11 cnn_eq_training_sym.py -m ASK -o 2
nice -n 5 python3.11 cnn_eq_training_sym.py -m ASK -o 4
nice -n 5 python3.11 cnn_eq_training_sym.py -m PAM -o 2
nice -n 5 python3.11 cnn_eq_training_sym.py -m PAM -o 4
nice -n 5 python3.11 cnn_eq_training_sym.py -m QAM -o 4

done

echo "All iterations completed."