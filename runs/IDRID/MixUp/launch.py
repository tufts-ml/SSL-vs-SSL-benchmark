import argparse
from pathlib import Path
import subprocess


launch_cmd = "pipenv run python -m ssl_bench.hyper_search"

slurm_args = [
    "--job-name=hyper_search",
    "--output=experiment_output_%j.log",
    "--ntasks=1",
    "--cpus-per-task=8",
    "--mem-per-cpu=2G",
    "--time=2:00:00",
    "--partition=hugheslab",
    "--gres=gpu:rtx_6000:1",
]

launch_args = [
    # method to use
    "--implementation MixUp",

    # IDRID config
    "--dataset_name IDRID",
    "--l_train_dataset_path /cluster/tufts/hugheslab/datasets/IDRID/Labels/train.csv",
    "--val_dataset_path /cluster/tufts/hugheslab/datasets/IDRID/Labels/validation.csv",
    "--test_dataset_path /cluster/tufts/hugheslab/datasets/IDRID/Labels/test.csv",
    "--root_dataset_path /cluster/tufts/hugheslab/datasets/IDRID/Images",

    # DataLoader config
    "--labeledtrain_batchsize 32",
    "--num_workers 8",

    # model config
    # "--freeze_backbone",
    "--use_pretrained",
    "--arch resnet18",

    # output location
    "--train_dir /cluster/tufts/hugheslab/sslbench/experiments/IDRID/MixUp/pytorch",

    # optimization config
    "--train_epoch 100",
    "--start_epoch 0",
    "--total_hour 1",
    "--optimizer_type Adam",
    "--lr_warmup_epochs 0",
    "--lr_schedule_type CosineLR",
    "--lr_cycle_epochs 100",  # should default to --train_epoch
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["print", "sbatch", "here"], default="print",
                        help="print: print config; sbatch: submit with sbatch; " +
                        "here: run locally without slurm")
    args = parser.parse_args()

    # ensure running in root of the repo so Pipenv and relative paths can be used
    cur_dir = Path(".")
    if cur_dir.absolute().name != "SSL-vs-SSL-benchmark":
        raise Exception(f"Should run in SSL-vs-SSL-benchmark, {cur_dir.absolute()} found instead")

    # construct the launch commands
    here_cmd = launch_cmd + " " + " ".join(launch_args)
    sbatch_cmd = f"sbatch {' '.join(slurm_args)} --wrap \"{here_cmd}\""

    # execute the requested mode
    if args.mode == "print":
        print("Printing current launch commands, to run use one of the following modes")
        print(f"sbatch:\n{sbatch_cmd}")
        print(f"here:\n{here_cmd}")
    elif args.mode == "sbatch":
        subprocess.run(sbatch_cmd, shell=True)
    elif args.mode == "here":
        subprocess.run(here_cmd, shell=True)
    else:
        raise Exception(f"Error handling mode {args.mode}")
