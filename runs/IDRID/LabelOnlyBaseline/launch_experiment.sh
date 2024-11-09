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

export resized_shape=384
export num_workers=20
export total_hour=25
export num_classes=5
export use_pretrained='False'
export patience=20

export implementation='LabelOnlyBaseline'

export resume='None'

#experiment setting
export dataset_name='IDRID'
export data_seed=2
export training_seed=0

export train_dir="/cluster/tufts/hugheslab/sslbench/experiments/$dataset_name/data_seed$data_seed/training_seed$training_seed/$implementation"

mkdir -p $train_dir

export script='src.LabelOnlyBaseline.LabelOnlyBaseline'


export arch='resnet18'
export train_epoch=200
export start_epoch=0


#data paths
export l_train_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Labels/train.csv'
export val_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Labels/validation.csv'
export test_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Labels/test.csv'
export root_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Images'

#shared config
export labeledtrain_batchsize=32 

#PL config, candidate hypers to search
export optimizer_type='Adam'


export lr_warmup_epochs=0
export lr_schedule_type='CosineLR'
export lr_cycle_epochs=$train_epoch


if [[ $ACTION_NAME == 'submit' ]]; then
    ## Submit the experiment with the correct resources for batch scheduler
    sbatch --time=0-02:00 --mem=30G --gres=gpu:rtx_6000:2 --cpus-per-task=30 -p hugheslab <./do_experiment.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Run the experiment interactively with the correct resources
    srun --time=0-02:00 --mem=30G --gres=gpu:rtx_6000:2 --cpus-per-task=30 -p hugheslab --pty bash ./do_experiment.slurm
fi