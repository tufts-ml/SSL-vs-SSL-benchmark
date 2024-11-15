import argparse

launch_args = {
    "--num_workers": 0,
    "--total_hour": 100,
    "--use_pretrained": 'False',
    "--patience": 20,
    "--implementation": 'PseudoLabeling',
    "--resume": 'last_checkpoint.pth.tar',
    "--dataset_name": 'TMED2',
    "--data_seed": 1,
    "--training_seed": 0,
    "--development_size": 'DEV56',
    # TODO fix the below 6 arguments
    "--train_dir": "$ROOT_PATH/experiments/$dataset_name/data_seed$data_seed/training_seed$training_seed/$implementation",
    "--script": "src.$implementation.$implementation",
    "--l_train_dataset_path": YOUR_PATH,
    "--u_train_dataset_path": YOUR_PATH,
    "--val_dataset_path": YOUR_PATH,
    "--test_dataset_path": YOUR_PATH,
    "--arch": 'wideresnet',
    "--train_epoch": 200,
    "--start_epoch": 0,
    "--labeledtrain_batchsize": 64, # default
    "--unlabeledtrain_batchsize": 64, # default
    "--em": 0, # default
    "--optimizer_type": 'Adam',
    "--threshold": 0.95,
    "--unlabeledloss_warmup_schedule_type": 'Linear',
    "--unlabeledloss_warmup_pos": 0.4, # FixMatch algo did not use unlabeled loss rampup schedule
    "--lr_warmup_epochs": 0,
    "--lr_schedule_type": 'CosineLR',
    "--lr_cycle_epochs": 200,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", default="submit", choices=["submit", "here", "print"],
                        help="submit to cluster, run on current node, or print run command")
    return parser.parse_args()


if __name__ == "__main__":
    # TODO make train_dir
    # TODO implement command behavior
    pass
