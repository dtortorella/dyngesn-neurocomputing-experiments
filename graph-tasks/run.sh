#!/bin/bash
for dataset in dblp_ct1 facebook_ct1 highschool_ct1 infectious_ct1 mit_ct1 tumblr_ct1 dblp_ct2 facebook_ct2 highschool_ct2 infectious_ct2 mit_ct2 tumblr_ct2; do
echo
echo $dataset
python dyngesn-tgk-datasets.py --dataset $dataset --root /tmp --device $1 --layers 1 2 3 4 5 6 --units 1 2 4 8 16 --sigma 0.1 0.3 0.5 0.7 0.9 1.0 1.1 1.3 1.5 1.7 1.9 2.0 2.1 2.3 2.5 --leakage 0.01 0.05 0.1 0.3 0.5 0.7 0.9 1.0 --ld 1e0 1e-1 1e-2 1e-3 1e-4 1e-5 --trials 200
done

