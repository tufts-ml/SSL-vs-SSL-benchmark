# SSL-vs-SSL-benchmark
Code for benchmark comparing self-supervised and semi-supervised deep classifiers for medical images


# Setup
### Prepare datasets
- TissueMNIST and PathMNIST: please visit https://zenodo.org/record/6496656
- TMED2: please visit https://TMED.cs.tufts.edu and follow the instruction.


### Install Anaconda
Follow the instructions here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Environment
packages needed are specified in environment.yml

# Running experiments
### Define the environment variable
```export ROOT_PATH="paths to this repo" ```
(e.g., '/ab/cd/SSL-vs-SSL-benchmark', then do export ROOT_PATH = '/ab/cd/SSL-vs-SSL-benchmark')

### Example
For example if you want to run Mean Teacher with Fix-A-Step for CIFAR-10 400labels/class , go to [runs/TissueMNIST/FixMatch/](runs/TissueMNIST/FixMatch/)

``` bash launch_experiment.sh run_here ```

### A note on reproducibility
While the focus of our paper is reproducibility, ultimately exact comparison to the results in our paper will be conflated by subtle differences such as the version of Pytorch etc (see https://pytorch.org/docs/stable/notes/randomness.html for more detail). We found in our experiment even with same random seed, result can vary sligtly between different runs (but usually less than 1%).


## Citing this work
TODO