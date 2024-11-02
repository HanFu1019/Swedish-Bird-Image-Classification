import requests
import os

# Prompt the user for the number of images per species
while True:
    try:
        image_count = int(input("Enter the number of images to download for each species: ").strip())
        if image_count > 0:
            break
        else:
            print("Please enter a positive integer for the number of images.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

# Create a directory for the dataset if it doesn't already exist
dataset_dir = 'Dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Read the species codes from the file
with open('ebird_species.txt', 'r') as file:
    species_list = file.read().splitlines()

# PUT YOUR HEADERS HERE
headers = {
    'User-Agent': '',
    'Accept': '',
    'Accept-Language': '',
    'Accept-Encoding': '',
    'Connection': '',
    'Cookie': '',
    'Upgrade-Insecure-Requests': '1'
}

# Loop through each species and download the CSV file
total_species = len(species_list)
for index, species in enumerate(species_list, start=1):
    # Create a directory for each species
    species_dir = os.path.join(dataset_dir, species)
    os.makedirs(species_dir, exist_ok=True)

    # Build the request URL for the CSV
    url = f'https://media.ebird.org/api/v2/export.csv?taxonCode={species}&mediaType=photo&sort=rating_rank_desc&birdOnly=true&count={image_count}'

    # Send a GET request to download the CSV
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an exception if there's an error

        # Save the CSV file in the respective species directory
        csv_path = os.path.join(species_dir, f'{species}.csv')
        with open(csv_path, 'wb') as csv_file:
            csv_file.write(response.content)

        # Show progress
        print(f"CSV for {species} downloaded and saved in {csv_path}. ({index}/{total_species} completed)")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download CSV for {species}: {e}")

print("All CSV files have been downloaded.")
