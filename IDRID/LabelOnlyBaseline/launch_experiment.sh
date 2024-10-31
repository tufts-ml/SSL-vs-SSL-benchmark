#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiment.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

export ROOT_PATH='/cluster/tufts/hugheslab/ljain01/SSL-vs-SSL-benchmark'
export PYTHONPATH="${PYTHONPATH}:$ROOT_PATH"

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

#export resized_shape=384
export num_workers=0
export total_hour=100
export num_classes=4
export use_pretrained='False'
export patience=20

export implementation='LabelOnlyBaseline'

export resume='last_checkpoint.pth.tar'

#experiment setting
export dataset_name='IDRID'
export data_seed=0
export training_seed=0

export train_dir="LABELONLYBASELINE"
mkdir -p $train_dir

export script='src.LabelOnlyBaseline.LabelOnlyBaseline'


export arch='resnet18'
export train_epoch=200 
export start_epoch=0


#data paths
export l_train_dataset_path='/cluster/tufts/hugheslab/ljain01/SSL-vs-SSL-benchmark/IDRID/train_data.npy'
export val_dataset_path='/cluster/tufts/hugheslab/ljain01/SSL-vs-SSL-benchmark/IDRID/val_data.npy'
export test_dataset_path='/cluster/tufts/hugheslab/ljain01/SSL-vs-SSL-benchmark/IDRID/test_data.npy'

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