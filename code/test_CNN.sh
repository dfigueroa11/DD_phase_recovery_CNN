#!/bin/bash

for i in {1..5}; do
echo "Running iteration $i"
#####################################
nice -n 5 python3.11 cnn_eq_training_sym.py "ASK" 2
nice -n 5 python3.11 cnn_eq_training_sym.py "ASK" 4
nice -n 5 python3.11 cnn_eq_training_sym.py "PAM" 2
nice -n 5 python3.11 cnn_eq_training_sym.py "PAM" 4
nice -n 5 python3.11 cnn_eq_training_sym.py "QAM" 4

done

echo "All iterations completed."