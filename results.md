# Results Table


## Full-Finetuning
| Family | Method | TMED2 | Chexpert | IDRID | CIFAR10 |
| ------ | ------ | ----- | -------- | ----- | ------- |
| Lab. only | CE | 32.81 (4/10) |78.57 (10/1)| 51.7651 (4/16) | 83.32 (2/16), TEST SET 13% |
| Lab. only | MixUp | 31.8541 (4/10) |78.00 (10/1)| 47.7351 (4/11) | |
| self      | Barlow Twins |30.8192 (9/17) |69.35 (10/1) | 36.1382 (4/18) | |
| self      | MoCo v3 | | | | |
| self      | SimCLR | |64.65 (10/15)| 26.5778 (5/2) | |
| semi      | MixMatch | |69.74 (10/15)|29.83 (10/20)| |
| semi      | FixMatch | |66.3 (12/17)| | |
| semi      | PseudoLab | |73.48 (11/11)| | |

## Linear Probing
| Family | Method | TMED2 | Chexpert | IDRID |
| ------ | --- | --- | --- | --- |
| Lab. only | CE | 30.6864 (4/10) |69.30 (10/1)| 45.7113 (4/16) |
| Lab. only | MixUp | 30.3466 (4/10) |68.85 (10/1)| 48.2996 (4/16) |
| self      | Barlow Twins | 30.5883 (9/17) |67.51 (10/1) | 34.0291 (4/18)|
| self      | MoCo v3 | | | |
| self      | SimCLR | |53.97 (10/16)| |
| semi      | MixMatch | |61.1 (10/15)| |
| semi      | FixMatch | | | |
| semi      | PseudoLab | |56.21 (10/29)| |

Table of (TODO balanced?) accuracy results and date when the experiment was completed in parentheses.
