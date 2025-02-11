#!/bin/bash
#SBATCH --job-name=hyper_search
#SBATCH --output=experiment_output_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00  # Adjust as needed
#SBATCH --partition=preempt

# Load the necessary modules (if required)
# module load python/3.8.5  # Example

# Activate the virtual environment
source /cluster/home/ljain01/.local/share/virtualenvs/SSL-vs-SSL-benchmark-za0Nt4Ck/bin/activate

# Export environment variables
export PYTHONPATH="${PYTHONPATH}:$ROOT_PATH"

# Running the Python script with parameters
python -m src.hyper_search \
    --method $method \
    --dataset_name $dataset_name \
    --l_train_dataset_path $l_train_dataset_path \
    --val_dataset_path $val_dataset_path \
    --test_dataset_path $test_dataset_path \
    --root_dataset_path $root_dataset_path \
    --labeledtrain_batchsize $labeledtrain_batchsize \
    --num_workers $num_workers \
    --arch $arch \
    --use_pretrained $use_pretrained \
    --train_epoch $train_epoch \
    --start_epoch $start_epoch \
    --total_hour $total_hour \
    --train_dir $train_dir \
    --lr_warmup_epochs $lr_warmup_epochs \
    --optimizer_type $optimizer_type \
    --lr_schedule_type $lr_schedule_type \
    --lr_cycle_epochs $lr_cycle_epochs
