config = {
'TissueMNIST': {'dataset_mean':(0.0988, 0.0988, 0.0988),
               'dataset_std':(0.0785, 0.0785, 0.0785),
               'image_size':28,
               'class_weights':[0.029, 0.195, 0.247, 0.1, 0.132, 0.195, 0.039, 0.063],
               'nimg_per_epoch':165466,
               'num_classes':8},

'PathMNIST': {'dataset_mean':(0.7403, 0.5310, 0.7062),
             'dataset_std':(0.0710, 0.1018, 0.0719),
             'image_size':28,
             'class_weights':[0.115, 0.113, 0.104, 0.104, 0.135, 0.089, 0.139, 0.115, 0.085],
             'nimg_per_epoch':89996,
             'num_classes':9},

'TMED2': {'dataset_mean':(0.0636, 0.0636, 0.0636),
         'dataset_std':(0.1426, 0.1426, 0.1426),
         'image_size':112,
         'class_weights':[0.137, 0.398, 0.192, 0.273],
         'nimg_per_epoch':355160,
         'num_classes':4},
}
