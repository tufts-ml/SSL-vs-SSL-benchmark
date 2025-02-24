#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'
export ROOT_PATH='/cluster/tufts/hugheslab/nfalic01/SSL-vs-SSL-benchmark'
export PYTHONPATH="${PYTHONPATH}:$ROOT_PATH"

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export num_workers=8
export total_hour=1
export patience=20

export implementation='LabelOnlyBaseline'

#experiment setting
export dataset_name='TMED2'

export train_dir="/cluster/tufts/hugheslab/sslbench/experiments/$dataset_name/lakshita_testing/$implementation"
mkdir -p $train_dir

export script="src.hyper_search"

export arch='resnet18'
export train_epoch=100
export start_epoch=0


#data paths
export l_train_dataset_path="/cluster/tufts/hugheslab/datasets/tmed/version2/labels_training.csv"
export u_train_dataset_path=""

export val_dataset_path="/cluster/tufts/hugheslab/datasets/tmed/version2/labels_val.csv"

export test_dataset_path="/cluster/tufts/hugheslab/datasets/tmed/version2/labels_test.csv"
export root_dataset_path="/cluster/tufts/hugheslab/datasets/tmed/version2"

#shared config
export labeledtrain_batchsize=512
export unlabeledtrain_batchsize=64


#PL config, candidate hypers to search
export optimizer_type='Adam'


export lr_warmup_epochs=0
export lr_schedule_type='CosineLR'
export lr_cycle_epochs=$train_epoch


if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch <./do_experiment.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash ./do_experiment.slurm
fi

