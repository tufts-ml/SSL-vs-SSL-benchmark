#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
export PYTHONPATH="${PYTHONPATH}:$ROOT_PATH"
python -m $script \
    --resized_shape $resized_shape\
    --num_workers $num_workers\
    --total_hour $total_hour\
    --use_pretrained $use_pretrained\
    --patience $patience\
    --dataset_name $dataset_name\
    --data_seed $data_seed\
    --training_seed $training_seed\
    --arch $arch\
    --train_epoch $train_epoch\
    --start_epoch $start_epoch\
    --resume $resume\
    --train_dir $train_dir\
    --l_train_dataset_path $l_train_dataset_path\
    --u_train_dataset_path $u_train_dataset_path\
    --val_dataset_path $val_dataset_path\
    --test_dataset_path $test_dataset_path\
    --labeledtrain_batchsize $labeledtrain_batchsize\
    --unlabeledtrain_batchsize $unlabeledtrain_batchsize\
    --em $em\
    --temperature $temperature\
    --mu $mu\
    --threshold $threshold\
    --lr_warmup_epochs $lr_warmup_epochs\
    --optimizer_type $optimizer_type\
    --lr_schedule_type $lr_schedule_type\
    --lr_cycle_epochs $lr_cycle_epochs\
    --unlabeledloss_warmup_schedule_type $unlabeledloss_warmup_schedule_type\
    --unlabeledloss_warmup_pos $unlabeledloss_warmup_pos\
    