import os
import pandas as pd

# Path to the dataset directory
dataset_dir = 'Dataset'

# Iterate over each folder in the dataset directory
for species_folder in os.listdir(dataset_dir):
    species_folder_path = os.path.join(dataset_dir, species_folder)

    # Ensure the item is a directory
    if os.path.isdir(species_folder_path):
        # Search for a CSV file in the folder
        csv_file = next((file for file in os.listdir(species_folder_path) if file.endswith('.csv')), None)
        if csv_file:
            csv_path = os.path.join(species_folder_path, csv_file)

            # Attempt to read the CSV file
            try:
                df = pd.read_csv(csv_path)

                # Extract the scientific name from the first row of the 'Scientific Name' column
                scientific_name = df['Scientific Name'].iloc[0]

                # Define the new path with the scientific name
                new_folder_path = os.path.join(dataset_dir, scientific_name)

                # Rename the folder if a folder with the scientific name does not already exist
                if not os.path.exists(new_folder_path):
                    os.rename(species_folder_path, new_folder_path)
                    print(f"Renamed '{species_folder}' to '{scientific_name}'")
                else:
                    print(f"Folder for '{scientific_name}' already exists. Skipping rename.")
            except (pd.errors.EmptyDataError, KeyError, IndexError) as e:
                print(f"Failed to process '{csv_file}' in '{species_folder}': {e}")
        else:
            print(f"No CSV file found in '{species_folder}'")
