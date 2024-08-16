#!/bin/bash
# Define shared parameters. The correct is False so we are working on the hallucination. 
gs=$1
hyp=$2
dataname=$3
testmode=$4
python constrained_reasoner.py --groundingsource "$gs" --hyp "$hyp" --dataname "$dataname" --category true --testmode "$testmode"
python constrained_reasoner.py --groundingsource "$gs" --hyp "$hyp" --dataname "$dataname" --category false --testmode "$testmode"
python constrained_reasoner.py --groundingsource "$gs" --hyp "$hyp" --dataname "$dataname" --category false --simple true --testmode "$testmode"
