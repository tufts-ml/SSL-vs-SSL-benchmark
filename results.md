# Results Table


## Full-Finetuning
| Family | Method | TMED2 | Chexpert | IDRID |
| ------ | ------ | ----- | -------- | ----- |
| Lab. only | CE | 32.81 (4/10) |79.12 (4/10)| 51.7651 (4/16) |
| Lab. only | MixUp | 31.8541 (4/10) |76.05 (4/10)| 47.7351 (4/11) |
| self      | Barlow Twins | | | 34.5416 (4/16)** |
| self      | MoCo v3 | | | |
| self      | SimCLR | | | |
| semi      | MixMatch | | | |
| semi      | FixMatch | | | |
| semi      | PseudoLab | | | |

## Linear Probing
| Family | Method | TMED2 | Chexpert | IDRID |
| ------ | --- | --- | --- | --- |
| Lab. only | CE | 30.6864 (4/10) |70.91 (4/10)| 45.7113 (4/16) |
| Lab. only | MixUp | 30.3466 (4/10) |73.13 (4/10)| 48.2996 (4/16) |
| self      | Barlow Twins | | | 36.9064 (4/17)|

Table of (TODO balanced?) accuracy results and date when the experiment was completed in parentheses.

** Current result is only with using 600 unlabeled images and training for 2 hours - preliminary result!
