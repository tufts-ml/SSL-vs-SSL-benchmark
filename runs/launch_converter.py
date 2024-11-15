import sys
from pathlib import Path


if __name__ == "__main__":
    input_file_path = Path(sys.argv[1])
    for line in open(input_file_path):
        if line.startswith("export"):
            split_line = line[6:].split("=")
            arg = split_line[0].strip()
            if "#" in split_line[1]:
                split_val = split_line[1].split("#")
                val = split_val[0].strip()
                comment = split_val[1].strip()
                print(f'"--{arg}": {val}, # {comment}')
            else:
                val = split_line[1].strip()
                print(f'"--{arg}": {val},')
