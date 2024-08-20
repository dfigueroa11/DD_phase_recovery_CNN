#!/bin/bash

for i in {1..4}; do
echo "Running iteration $i"
#####################################
python3.11 cnn_eq_training_phase_diff.py "ASK" 2
python3.11 cnn_eq_training_phase_diff.py "ASK" 4
python3.11 cnn_eq_training_phase_diff.py "QAM" 4
python3.11 cnn_eq_training_phase_diff.py "DDQAM" 4

#####################################
python3.11 cnn_eq_training_phase.py "ASK" 2
python3.11 cnn_eq_training_phase.py "ASK" 4
python3.11 cnn_eq_training_phase.py "QAM" 4
python3.11 cnn_eq_training_phase.py "DDQAM" 4

#####################################
python3.11 cnn_eq_training_phase_diff_mag_in.py "ASK" 4
python3.11 cnn_eq_training_phase_diff_mag_in.py "DDQAM" 8

#####################################
python3.11 cnn_eq_training_odd_samp.py "PAM" 2
python3.11 cnn_eq_training_odd_samp.py "PAM" 4
python3.11 cnn_eq_training_odd_samp.py "ASK" 4

#####################################
python3.11 cnn_eq_training_odd_samp_phase_in.py "ASK" 4

done

echo "All iterations completed."