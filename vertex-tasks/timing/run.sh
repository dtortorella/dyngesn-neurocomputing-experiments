#!/bin/bash
for dataset in chickenpox tennis pedalme wikimath; do
  echo $dataset
  python dyngesn-model.py --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-1h.py --model dcrnn --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-1h.py --model gconvgru --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-2h.py --model gconvlstm --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-2h.py --model gclstm --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-dygrae.py --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-0h.py --model egcnh --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-0h.py --model egcno --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-a3tgcn.py --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-1h.py --model tgcn --dataset $dataset --device $1 | tee -a $dataset.txt
  python e2e-models-0h.py --model mpnnlstm --dataset $dataset --device $1 | tee -a $dataset.txt
done

