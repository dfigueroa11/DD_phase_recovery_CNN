#!/bin/bash

for i in {1,2,3,4}
do
    python3.11 tvrnn_eq_training.py -m ASK -o 2
    python3.11 tvrnn_eq_training.py -m ASK -o 4
    python3.11 tvrnn_eq_training.py -m PAM -o 2
    python3.11 tvrnn_eq_training.py -m PAM -o 4
    python3.11 tvrnn_eq_training.py -m QAM -o 4
    wait
done
echo "ALL DONE"