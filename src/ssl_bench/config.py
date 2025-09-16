from scipy.stats import loguniform
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from ssl_bench.utils.apply_clahe import apply_clahe

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
              'nimg_per_epoch': 17270,
              'num_classes': 4},

    'IDRID': {
        'dataset_mean': (0.4250, 0.2092, 0.0707),
        'dataset_std': (0.3227, 0.1689, 0.0850),
        'image_size': 384,
        'class_weights': [0.0756, 0.5064, 0.0745, 0.1369, 0.2067],
        'nimg_per_epoch': 206,
        'num_classes': 5},

    'CheXpert2': {'dataset_mean': (0.5064, 0.5064, 0.5064),
                 'dataset_std': (0.2894, 0.2894, 0.2894),
                 'image_size': (320, 390),
                 'class_weights': [0.3027, 0.6973],
                 'nimg_per_epoch': 3500,
                 'num_classes': 2},

    'CheXpertEffusion': {'dataset_mean': (0.5064, 0.5064, 0.5064),
                  'dataset_std': (0.2894, 0.2894, 0.2894),
                  'image_size': (320, 390),
                  'class_weights': [0.3027, 0.6973],
                  'nimg_per_epoch': 3500,
                  'num_classes': 2},
}

class TransformTwice:
    def __init__(self, transform_fn):
        self.transform_fn = transform_fn

    def __call__(self, x):
        out1 = self.transform_fn(x)
        out2 = self.transform_fn(x)

        return out1, out2

class TransformFixMatch:
    def __init__(self, weak_transform, strong_transform):
        self.weak = weak_transform
        self.strong = strong_transform

    def __call__(self, x):
        return self.weak(x), self.strong(x)


def get_transformations(args):
    dataset_name = args.dataset_name
    method = args.implementation

    dataset_mean = dataset_configs[dataset_name]['dataset_mean']
    dataset_std = dataset_configs[dataset_name]['dataset_std']
    image_dim = dataset_configs[dataset_name]['image_size']
    
    # dealing with issue in transformation dictionary for methods
    # expecting a single value when image_size is a tuple
    if isinstance(image_dim, tuple):
        img_size = image_dim[0]
    else:
        img_size = image_dim

    # training from scratch 
    dataset_base_transformations = {
        'TMED2': {
            "l_train": transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Lambda(apply_clahe),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=img_size,
                                        padding=int(img_size*0.125),
                                        padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ]),
            "u_train": None,
            "val": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(400),
                transforms.CenterCrop(320),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ]),
            "test": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(400),
                transforms.CenterCrop(320),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ])
        },
        'IDRID': {
            "l_train": transforms.Compose([
                transforms.Resize(size=img_size),
                transforms.Lambda(apply_clahe),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=img_size,
                                    padding=int(img_size*0.125),
                                    padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ]),
            "u_train": None,
            "val": transforms.Compose([
                transforms.Resize(size=img_size),
                transforms.Lambda(apply_clahe),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ]),
            "test": transforms.Compose([
                transforms.Resize(size=img_size),
                transforms.Lambda(apply_clahe),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ])
        }, 
        'CheXpert2': {
            "l_train": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(400),
                transforms.CenterCrop(320),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ]),
            "u_train": None,
            "val": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(400),
                transforms.CenterCrop(320),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ]),
            "test": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(400),
                transforms.CenterCrop(320),
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_std)
            ])
        },
        'Pretrained': {
            "l_train": transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                ResNet18_Weights.IMAGENET1K_V1.transforms(),
            ]),
            "u_train": None,
            "val": transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                ResNet18_Weights.IMAGENET1K_V1.transforms(),
            ]),
            "test": transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                ResNet18_Weights.IMAGENET1K_V1.transforms(),
            ])
        }
    }


    method_transformations = {
        "LabelOnlyBaseline": {
            "l_train": None,
            "u_train": None,
            "val": None,
            "test": None
        },
        "MixUp": {
            "l_train": None,
            "u_train": None,
            "val": None,
            "test": None
        },
        "BarlowTwins": {
            "l_train": None,
            "u_train": TransformTwice(dataset_base_transformations[dataset_name]["l_train"]),
            "val": None,
            "test": None
        }
    }

    if dataset_name not in dataset_base_transformations:
        raise NotImplementedError(f"Must add in base transformations for dataset {dataset_name}")
    
    pretrained = args.use_pretrained
    data_partitions = list(dataset_base_transformations['Pretrained'].keys())
    transformation_key = "Pretrained" if pretrained else dataset_name

    if (method in method_transformations):
        for split in data_partitions:
            if method_transformations[method][split] != None:
                dataset_base_transformations[transformation_key][split] = method[split]
    else:
        raise NotImplementedError(f"Must add in transformations for method: {method}")
    
    l_train_transform, u_train_transform, val_transform, test_transform = list(dataset_base_transformations[transformation_key].values())
    return l_train_transform, u_train_transform, val_transform, test_transform


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
    }),
    'MixUp': HyperparamSpace({
        "lr": loguniform(1e-5, 1e-2),
        "wd": loguniform(1e-6, 1e-3),
        "alpha": loguniform(1e-1, 10),
    }),
    'FixMatch': HyperparamSpace({
        "lr": loguniform(1e-5, 1e-2),
        "wd": loguniform(1e-6, 1e-3),
        "conf_threshold": loguniform(0.7, 0.95),
    }),
    'BarlowTwins': HyperparamSpace({
        "lr": loguniform(1e-5, 1e-2),
        "wd": loguniform(1e-6, 1e-3),
        "lambd": loguniform(1e-3, 1e-2),
    }),
}
