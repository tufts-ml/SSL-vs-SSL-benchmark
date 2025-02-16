from scipy.stats import loguniform

dataset_configs = {
    'TissueMNIST': {'dataset_mean': (0.0988, 0.0988, 0.0988),
                    'dataset_std': (0.0785, 0.0785, 0.0785),
                    'image_size': 28,
                    'class_weights': [0.029, 0.195, 0.247, 0.1, 0.132, 0.195,
                                      0.039, 0.063],
                    'nimg_per_epoch': 165466,
                    'num_classes': 8},

    'PathMNIST': {'dataset_mean': (0.7403, 0.5310, 0.7062),
                  'dataset_std': (0.0710, 0.1018, 0.0719),
                  'image_size': 28,
                  'class_weights': [0.115, 0.113, 0.104, 0.104, 0.135, 0.089,
                                    0.139, 0.115, 0.085],
                  'nimg_per_epoch': 89996,
                  'num_classes': 9},

    'TMED2': {'dataset_mean': (0.0636, 0.0636, 0.0636),
              'dataset_std': (0.1426, 0.1426, 0.1426),
              'image_size': 112,
              'class_weights': [0.137, 0.398, 0.192, 0.273],
              'nimg_per_epoch': 355160,
              'num_classes': 4},

    'IDRID': {
        'dataset_mean': (0.4250, 0.2092, 0.0707),
        'dataset_std': (0.3227, 0.1689, 0.0850),
        'image_size': 384,
        'class_weights': [0.0756, 0.5064, 0.0745, 0.1369, 0.2067],
        'nimg_per_epoch': 206,
        'num_classes': 5},

    'CheXpert': {'dataset_mean': (0.5064, 0.5064, 0.5064),
                 'dataset_std': (0.2894, 0.2894, 0.2894),
                 'image_size': (320, 390),
                 'class_weights': [0.3027, 0.6973],
                 'nimg_per_epoch': 3500,
                 'num_classes': 2},

    'CIFAR100': {'dataset_mean': (0.49139968, 0.48215827, 0.44653124),
                 'dataset_std': (0.24703233, 0.24348505, 0.26158768),
                 'image_size': 32,
                 'class_weights': [0.01] * 100,
                 'nimg_per_epoch': 50000,
                 'num_classes': 100
                 },
}


class HyperparamSpace():
    def __init__(self, hyperparam_dist_dict: dict):
        """Hyperparameter space to sample hyperparameter values from randomly

        Args:
            hyperparam_dist_dict (dict): maps hyperparameter (str) to sampling distribution
                                         (scipy.stats class)
        """
        self.hyperparam_dist_dict = hyperparam_dist_dict

    def rvs(self, size=1, random_state=None):
        return {hyperparam: (dist.rvs(size, random_state).item() if size == 1
                             else dist.rvs(size, random_state))
                for hyperparam, dist in self.hyperparam_dist_dict.items()}


method_configs = {
    'LabelOnlyBaseline': HyperparamSpace({
        "lr": loguniform(1e-5, 1e-2),
        "wd": loguniform(1e-6, 1e-3),
    })
}
