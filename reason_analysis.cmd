set gs=%1
set hyp=%2
set dataname=%3
set testmode=%4
python constrained_reasoner.py --groundingsource %gs% --hyp %hyp% --dataname %dataname% --category true --testmode %testmode%  
python constrained_reasoner.py --groundingsource %gs% --hyp %hyp% --dataname %dataname% --category false --testmode %testmode%  
python analysis.py --dataname %dataname%