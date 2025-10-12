import pandas as pd

input_file = "/cluster/tufts/hugheslab/datasets/CheXpert/train.csv"
output_file = "/cluster/tufts/hugheslab/datasets/CheXpert/train_data_effusion.csv"

df = pd.read_csv(input_file)

filtered_df = df[['Path', 'Pleural Effusion']].copy()

filtered_df['Pleural Effusion'] = filtered_df['Pleural Effusion'].fillna(0)
filtered_df = filtered_df[filtered_df['Pleural Effusion'] != -1]

filtered_df.to_csv(output_file, index=False)

print(f"Filtered dataset saved to: {output_file}")
