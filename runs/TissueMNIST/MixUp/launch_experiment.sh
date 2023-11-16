#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export num_workers=0
export total_hour=50
export use_pretrained='False'
export patience=20

export implementation='MixUp'

export resume='last_checkpoint.pth.tar'

#experiment setting
export dataset_name='TissueMNIST'
export data_seed=0
export training_seed=4

export train_dir="$ROOT_PATH/experiments/$dataset_name/data_seed$data_seed/training_seed$training_seed/$implementation"
mkdir -p $train_dir

export script="src.$implementation.$implementation"


export arch='resnet18'
export train_epoch=200 
export start_epoch=0


#data paths
export l_train_dataset_path=YOUR_PATH


export val_dataset_path=YOUR_PATH

export test_dataset_path=YOUR_PATH


#shared config
export labeledtrain_batchsize=64 #default


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


