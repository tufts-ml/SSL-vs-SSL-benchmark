#!/bin/bash

# Usage
# -----
# $ bash launch_experiment.sh ACTION_NAME
#
# where ACTION_NAME is either 'list', 'submit', or 'run_here'

export ROOT_PATH='/cluster/tufts/hugheslab/ljain01/SSL-vs-SSL-benchmark'
export PYTHONPATH="${PYTHONPATH}:$ROOT_PATH"

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

# Set up environment variables
export resized_shape=384
export num_workers=8
export total_hour=25
export num_classes=5
export use_pretrained=true
export freeze_backbone=true
export patience=20
export implementation='LabelOnlyBaseline'

export resume='last_checkpoint.pth.tar'

# Experiment setting
export dataset_name='IDRID'

export train_dir="/cluster/tufts/hugheslab/sslbench/experiments/$dataset_name/$implementation/eval_test/"

mkdir -p $train_dir

export script="src.hyper_search"  # Ensure this is set correctly

export arch='resnet18'
export train_epoch=200 
export start_epoch=0

# Data paths
export l_train_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Labels/train.csv'
export val_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Labels/validation.csv'
export test_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Labels/test.csv'
export root_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Images'

# Shared config
export labeledtrain_batchsize=32

# PL config, candidate hypers to search
export optimizer_type='Adam'
export lr_warmup_epochs=0
export lr_schedule_type='CosineLR'
export lr_cycle_epochs=$train_epoch

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch --export=ALL ./do_experiment.slurm  # Fixed from `< ./do_experiment.slurm`
    
elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to run the script interactively
    bash ./do_experiment.slurm
fi
