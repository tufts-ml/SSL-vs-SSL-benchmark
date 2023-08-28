# SSL-vs-SSL-benchmark
Code for benchmark comparing self-supervised and semi-supervised deep classifiers for medical images


# Setup
### Prepare datasets
- TissueMNIST and PathMNIST: please visit https://zenodo.org/record/6496656
- AIROGS: please visit: https://zenodo.org/record/5793241 
- TMED2: please visit https://TMED.cs.tufts.edu and follow the instruction. We use the split1 in the released data.

For the exact data we use in the experiment (train/val/test/unlabeled split), you can download it from the box folder https://tufts.box.com/s/wxmd8a1gbkmau57n05vquu3e5sqbpgls


### Install Anaconda
Follow the instructions here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Environment
packages needed are specified in environment.yml (TODO)

# Running experiments
### Define the environment variable
```export ROOT_PATH="paths to this repo" ```
(e.g., '/ab/cd/SSL-vs-SSL-benchmark', then do export ROOT_PATH = '/ab/cd/SSL-vs-SSL-benchmark')


### Example
For example if you want to run FixMatch on TissueMNIST to reproduce Figure 1(a) and Figure A.2(a), go to [runs/TissueMNIST/FixMatch/](runs/TissueMNIST/FixMatch/)
``` bash launch_experiment.sh run_here ```

Note that you will need to edit the paths to dataset in the launch_experiment.sh file.
### A note on reproducibility
While the focus of our paper is reproducibility, ultimately exact comparison to the results in our paper will be conflated by subtle differences such as the version of Pytorch etc (see https://pytorch.org/docs/stable/notes/randomness.html for more detail).

### Acknowledgement
This repository builds upon the pytorch-consistency-regularization [https://github.com/perrying/pytorch-consistency-regularization]. Thanks for sharing the great code bases!

## Reference
TODO


