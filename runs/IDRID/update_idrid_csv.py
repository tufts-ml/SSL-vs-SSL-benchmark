import pandas as pd


def modify_csv(input_csv, output_csv, directory_prefix=''):
    # Read the CSV file
    df = pd.read_csv(input_csv)
   
    # Keep only the first two columns
    df = df.iloc[:, :2]
    
    # Modify the first column to add 'Training/' at the beginning and '.jpg' at the end
    df.iloc[:, 0] = directory_prefix + df.iloc[:, 0].astype(str) + '.jpg'
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    
    
# Modify the CSV file for the training split
input_csv = '/cluster/tufts/hugheslab/datasets/IDRID/Labels/training_split.csv' 
output_csv = '/cluster/tufts/hugheslab/datasets/IDRID/Labels/train.csv'
modify_csv(input_csv, output_csv, directory_prefix='Training/')

# Modify the CSV file for the validation split
input_csv = '/cluster/tufts/hugheslab/datasets/IDRID/Labels/validation_split.csv' 
output_csv = '/cluster/tufts/hugheslab/datasets/IDRID/Labels/validation.csv'
modify_csv(input_csv, output_csv, directory_prefix='Training/')

# Modify the CSV file for the test split
input_csv = '/cluster/tufts/hugheslab/datasets/IDRID/Labels/Testing.csv'
output_csv = '/cluster/tufts/hugheslab/datasets/IDRID/Labels/test.csv'
modify_csv(input_csv, output_csv, directory_prefix='Testing/')