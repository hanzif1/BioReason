# import openml
import openml

# download dataset with DATASET_ID. Check Dataset detail page for DATASET_ID
dataset = openml.datasets.get_dataset('Birds', download_data=True, download_all_files=True)

# display dataset info
print(dataset.name)