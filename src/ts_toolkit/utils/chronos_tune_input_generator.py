import os
from datetime import datetime
import datasets
import numpy as np
from tqdm import tqdm

def chronos_input_generator(dataset_path, dataset_list_level1, dataset_list_level2, val_size=0.1):
    """Generates training and validation inputs for Chronos2 from the SimpleTSbench datasets."""
    # Path to the SimpleTSbench datasets
    train_dataset = []
    valid_dataset = []
    if not dataset_list_level1:
        dataset_list_level1 = os.listdir(dataset_path)
    # Iterate through dataset types (multivariate, multivariate_lagged)
    for dataset_type in dataset_list_level1: #tqdm(os.listdir(simpletsbench_path)):
        dataset_type_path = os.path.join(dataset_path, dataset_type)
        
        if not os.path.isdir(dataset_type_path):
            continue
        if not dataset_list_level2:
            dataset_list_level2 = os.listdir(dataset_type_path)
        # Iterate through trend types (constant, exponential, linear, etc.)
        for trend_type in tqdm(dataset_list_level2):
            trend_path = os.path.join(dataset_type_path, trend_type)
            
            if not os.path.isdir(trend_path):
                continue
            
            dataset_name = f"{dataset_type}/{trend_type}"
            # print(f"Processing: {dataset_name}")
            
            try:
                # Load the dataset using Hugging Face datasets library
                hf_dataset = datasets.load_from_disk(trend_path).with_format("numpy")
                
                # print(f"Loaded {len(hf_dataset)} rows from {dataset_name}")
                
                if len(hf_dataset) == 0:
                    print(f"Warning: Empty dataset in {dataset_name}")
                    continue
                
                # Iterate through each time series in the dataset
                for idx, row in tqdm(enumerate(hf_dataset), desc=f"{dataset_type}/{trend_type}"):
                    item_id = row['item_id']
                    
                    # Extract target - it's already in numpy format due to with_format("numpy")
                    target_data = row['target']
                    
                    if target_data is None or len(target_data) == 0:
                        continue
                    
                    # Convert to numpy array if not already
                    target_array = np.array(target_data)
                    
                    # If multivariate (2D array), transpose to (num_timesteps, num_variables)
                    if target_array.ndim == 2:
                        target_array = target_array.T
                    if target_array.ndim == 1:
                        target_array = target_array.reshape(-1, 1)
                    data_input = {"target": target_array[:, 0]}
                    if target_array.shape[1] > 1:
                        data_input.update({
                            "past_covariates": {f'feat_{i}': target_array[:, i] for i in range(1, target_array.shape[1])}
                        })
                    if idx < len(hf_dataset) * (1 - val_size):
                        train_dataset.append(data_input)
                    else:
                        valid_dataset.append(data_input)
                    
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                # continue

    print(f"\nPrepared training inputs for {len(train_dataset)} training time series and {len(valid_dataset)} validation time series.")

    # Show a sample of what we loaded
    if train_dataset:
        sample = train_dataset[0]
        print(f"\nSample time series:")
        print(f"  Target shape: {sample['target'].shape}")
    else:
        print("\nWarning: No training inputs were loaded!")
    return train_dataset, valid_dataset