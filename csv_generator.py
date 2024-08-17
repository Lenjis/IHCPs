import os
import pandas as pd # type: ignore

def csv_generator(env_var):
    # Define the input and output paths
    input_path = os.path.join(env_var, 'input')
    output_path = os.path.join(env_var, 'output')

    # Initialize the data list
    data = []

    # Iterate over folders in the input path
    for folder_name in sorted(os.listdir(input_path)):
        input_folder = os.path.join(input_path, folder_name)
        output_folder = os.path.join(output_path, folder_name)

        # Check if both input and output folders exist
        if os.path.isdir(input_folder) and os.path.isdir(output_folder):
            data.append({
                'folder_name': folder_name,
                'input_folder': input_folder,
                'output_folder': output_folder
            })

    # Convert the list to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(env_var, 'dataset_info.csv'), index=False)

# Example usage:
env_var = os.path.normpath('/home/linux/IHCPs/dataset_pm1000_sin')
csv_generator(env_var)
