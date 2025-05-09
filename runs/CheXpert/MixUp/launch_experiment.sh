#!/bin/bash

# Usage
# -----
# $ bash launch_experiment.sh ACTION_NAME
#
# where ACTION_NAME is either 'list', 'submit', or 'run_here'

export ROOT_PATH='/cluster/tufts/hugheslab/abaran03/SSL-vs-SSL-benchmark'
export PYTHONPATH="${PYTHONPATH}:$ROOT_PATH/src"

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

# Set up environment variables
export resized_shape=384
export num_workers=4
export total_hour=20
export num_classes=2
export use_pretrained=""
export patience=200

export method='MixUp'
export resume='last_checkpoint.pth.tar'

# Experiment setting
export dataset_name='CheXpertEffusion'
export data_seed=0
export training_seed=0

export train_dir="/cluster/tufts/hugheslab/sslbench/experiments/$dataset_name/MixUp20effusion/data_seed$data_seed/training_seed$training_seed/$implementation/pretrained$use_pretrained"
echo "Training directory: $train_dir"

mkdir -p $train_dir

export script="ssl_bench.hyper_search"

export arch='resnet18'
export train_epoch=200 
export start_epoch=0

# Data paths
# export l_train_dataset_path='/cluster/tufts/hugheslab/datasets/chexpert_sample_data/train_data.csv'
# export val_dataset_path='/cluster/tufts/hugheslab/datasets/chexpert_sample_data/val_data.csv'
# export test_dataset_path='/cluster/tufts/hugheslab/datasets/chexpert_sample_data/test_data.csv'
# export root_dataset_path='/cluster/tufts/hugheslab/datasets/chexpert_sample_data/images'
export l_train_dataset_path='/cluster/tufts/hugheslab/datasets/chexpert_sample_data/train_data_effusion.csv'
export val_dataset_path='/cluster/tufts/hugheslab/datasets/chexpert_sample_data/val_data_effusion.csv'
export test_dataset_path='/cluster/tufts/hugheslab/datasets/chexpert_sample_data/test_data_effusion.csv'
export root_dataset_path='/cluster/tufts/hugheslab/datasets/chexpert_sample_data/images'




# Shared config
export labeledtrain_batchsize=64

# PL config, candidate hypers to search
export optimizer_type='Adam'
export lr_warmup_epochs=0
export lr_schedule_type='CosineLR'
export lr_cycle_epochs=$train_epoch

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch ./do_experiment.slurm  # Fixed from `< ./do_experiment.slurm`
    
elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to run the script interactively
    bash ./do_experiment.slurm
fi
