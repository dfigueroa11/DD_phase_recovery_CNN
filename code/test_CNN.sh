#!/bin/bash

nice -n 5 python3.11 cnn_eq_training_complexity_sweep.py -m ASK -o 2 -l 5 &
nice -n 5 python3.11 cnn_eq_training_complexity_sweep.py -m ASK -o 4 -l 5 &
nice -n 5 python3.11 cnn_eq_training_complexity_sweep.py -m PAM -o 2 -l 1 &
wait
nice -n 5 python3.11 cnn_eq_training_complexity_sweep.py -m PAM -o 4 -l 1 &
nice -n 5 python3.11 cnn_eq_training_complexity_sweep.py -m QAM -o 4 -l 0 &
wait
echo "ALL DONE"