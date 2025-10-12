#!/bin/bash
#SBATCH --job-name=hyper_search
#SBATCH --output=experiment_output_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=50:00:00
#SBATCH --partition=hugheslab
#SBATCH --gres=gpu:rtx_6000:1

# Load the necessary modules (if required)
# module load python/3.8.5  # Example

# Activate the virtual environment
source /cluster/home/abaran03/.local/share/virtualenvs/SSL-vs-SSL-benchmark-P85lwjoW/bin/activate

# Export environment variables
export ROOT_PATH='/cluster/tufts/hugheslab/abaran03/SSL-vs-SSL-benchmark'
export PYTHONPATH="${PYTHONPATH}:$ROOT_PATH/src"


# Running the Python script with parameters
python3 -m ssl_bench.hyper_search \
    --implementation $method \
    --dataset_name $dataset_name \
    --l_train_dataset_path $l_train_dataset_path \
    --val_dataset_path $val_dataset_path \
    --test_dataset_path $test_dataset_path \
    --l_root_dataset_path $l_root_dataset_path \
    --labeledtrain_batchsize $labeledtrain_batchsize \
    --num_workers $num_workers \
    --arch $arch \
    --train_epoch $train_epoch \
    --start_epoch $start_epoch \
    --total_hour $total_hour \
    --train_dir $train_dir \
    --lr_warmup_epochs $lr_warmup_epochs \
    --optimizer_type $optimizer_type \
    --lr_schedule_type $lr_schedule_type \
    --lr_cycle_epochs $lr_cycle_epochs \
    --use_pretrained \
    --freeze_backbone


