import os
import pickle
import glob
import pickle
import matplotlib.pyplot as plt

global_stats_dir = 'GLOBAL_STATS_FILE_PATH'
pickle_files = [file for file in os.listdir(global_stats_dir) if file.endswith('.pkl')]


with open('global_stats/hypercombo_iteratethrough_list.pkl', 'rb') as file:
    hypercombo_list = pickle.load(file)

with open('global_stats/global_best_test_raw_acc_at_val_list_parallel.pkl', 'rb') as file:
    test_accuracy_data = pickle.load(file)

with open('global_stats/global_best_val_raw_acc_list_parallel.pkl', 'rb') as file:
    val_accuracy_data = pickle.load(file)

# Set up the figure for subplots
num_plots = len(hypercombo_list)
fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)

# Iterate through each hyperparameter combination
for i, (hypercombo, test_acc, val_acc) in enumerate(zip(hypercombo_list, test_accuracy_data, val_accuracy_data)):
    lr = hypercombo['lr']
    wd = hypercombo['wd']
    
    axs[i].plot(test_acc, label='Test Accuracy', color='blue')
    axs[i].plot(val_acc, label='Validation Accuracy', color='orange')
    
    axs[i].set_title(f'Learning Rate: {lr}, Weight Decay: {wd}')
    axs[i].set_ylabel('Accuracy')
    axs[i].legend()
    
# Set common x-label
axs[-1].set_xlabel('Epochs')

plt.tight_layout()
plt.show()

# Save the resulting image
plt.savefig('global_stats/YOUR_FILE_NAME.png')