#!/bin/bash

python3.11 cnn_eq_training_phase.py "ASK" 2

python3.11 cnn_eq_training_phase.py "ASK" 4

python3.11 cnn_eq_training_phase.py "QAM" 4

python3.11 cnn_eq_training_phase.py "DDQAM" 4

python3.11 cnn_eq_training_mag.py "PAM" 2

python3.11 cnn_eq_training_mag.py "PAM" 4

python3.11 cnn_eq_training_mag.py "ASK" 4
