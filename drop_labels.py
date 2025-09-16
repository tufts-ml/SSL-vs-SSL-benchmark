# drop_labels.py
input_path = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/train_data_effusion.csv"
output_path = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/unlabeled_tr.csv"

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    header = infile.readline()
    outfile.write("Path\n")  # Write new header

    for line in infile:
        path = line.strip().split(",")[0]
        outfile.write(path + "\n")
