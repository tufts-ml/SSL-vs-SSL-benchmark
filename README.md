# SSL-vs-SSL-benchmark
This is the code for benchmark comparing self-supervised and semi-supervised deep classifiers for medical images

# Supplementary Materials
Here we provide the Supplementary Materials[Appendix.pdf/] of our benchmark. 
The Supplement includes following sections to describe the experiments and analysis in more details.
- Code and Data Splits for Reproducibility
- Dataset Details
- Additional Results
- Algorithms Details
- Additional Analysis


# Setup
## Prepare datasets
- TissueMNIST and PathMNIST: please visit https://zenodo.org/record/6496656
- AIROGS: please visit: https://zenodo.org/record/5793241 
- TMED2: please visit https://TMED.cs.tufts.edu and follow the instruction. We use the split1 in the released data.

## Pipenv Installation

The following instructions should be done with Python 3.10 to create a Pipenv with all required packages installed. If you do not have Pipenv installed, run the following:
```
pip install pipenv
```
The dependencies can be installed within a Pipenv with the following commands:
```
pipenv install --categories "packages torch_cpu"
```
PyTorch may require different versions depending on the machine it is running on. The default command is for non-CUDA machines while swapping `torch_cpu` for `torch_cu117` installs PyTorch for CUDA 11.7. If a non-default version of PyTorch is required then generate the appropriate Pip command on the [PyTorch website](https://pytorch.org/get-started/locally/) then run it within the Pipenv by prepending ```pipenv run``` to it.

# Running experiments
## Define the environment variable
```export ROOT_PATH="paths to this repo" ```
(e.g., '/ab/cd/SSL-vs-SSL-benchmark', then do export ROOT_PATH = '/ab/cd/SSL-vs-SSL-benchmark')


## Example
For example if you want to run FixMatch on TissueMNIST to reproduce Figure 1(a) and Figure A.2(a), go to [runs/TissueMNIST/FixMatch/](runs/TissueMNIST/FixMatch/)
``` bash launch_experiment.sh run_here ```

Note that you will need to edit the paths to dataset in the launch_experiment.sh file.
## A note on reproducibility
While the focus of our paper is reproducibility, ultimately exact comparison to the results in our paper will be conflated by subtle differences such as the version of Pytorch etc (see https://pytorch.org/docs/stable/notes/randomness.html for more detail).

## Acknowledgement
This repository builds upon the public repo pytorch-consistency-regularization[https://github.com/perrying/pytorch-consistency-regularization]. Thanks for sharing the great code bases!

## Reference
TODO


