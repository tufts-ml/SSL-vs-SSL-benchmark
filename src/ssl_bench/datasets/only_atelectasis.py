import pandas as pd

input_file = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/val_data.csv"
output_file = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/val_data_atelectasis.csv"

df = pd.read_csv(input_file)

filtered_df = df[['Path', 'Atelectasis']]

filtered_df['Atelectasis'] = filtered_df['Atelectasis'].fillna(0)
filtered_df['Atelectasis'] = filtered_df['Atelectasis'].replace(-1, 1)

filtered_df.to_csv(output_file, index=False)

print(f"Filtered dataset saved to: {output_file}")
