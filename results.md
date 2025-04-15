# Results Table


## Full-Finetuning
| Family | Method | TMED2 | Chexpert | IDRID |
| ------ | ------ | ----- | -------- | ----- |
| Lab. only | CE | 32.81 (4/10) |79.12 (4/10)| 42.5186 (TODO) |
| Lab. only | MixUp | 31.8541 (4/10) |76.05 (4/10)| 47.7351 (4/11) |
| self      | Barlow Twins | | | |
| self      | MoCo v3 | | | |
| self      | SimCLR | | | |
| semi      | MixMatch | | | |
| semi      | FixMatch | | | |
| semi      | PseudoLab | | | |

## Linear Probing
| Method | TMED2 | Chexpert | IDRID |
| --- | --- | --- | --- |
| CE | 30.6864 (4/10) |70.91 (4/10)| 45.9161 (TODO) |
| MixUp | 30.3466 (4/10) |73.13 (4/10)| 43.3290 (TODO) |
| Barlow Twins | | | |

Table of (TODO balanced?) accuracy results and date when the experiment was completed in parentheses.
