import requests
from bs4 import BeautifulSoup

country_code = input("Enter the country code (e.g., 'SE' for Sweden, 'US' for USA, 'AQ' for Antarctica): ").strip().upper()

url = f'https://ebird.org/region/{country_code}/bird-list'

try:
    
    response = requests.get(url)
    response.raise_for_status()

    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the region name from the page title
    title_tag = soup.find('title')
    region_name = title_tag.text.split(' - ')[1] if title_tag else 'Unknown Region'

    
    species_links = [
        link['href'].split('/')[-2]
        for link in soup.find_all('a', href=True)
        if '/species/' in link['href'] and link['href'].endswith(f'/{country_code}')
    ]

    # Sort and write the species identifiers to a text file
    species_links.sort()
    with open('ebird_species.txt', 'w') as file:
        for species in species_links:
            file.write(species + '\n')

    # Summary information
    print("---------------------------")
    print("Region:", region_name)
    print("Country code:", country_code)
    print("Number of species found:", len(species_links))
    print("File name: ebird_species.txt")
    print("---------------------------")

except requests.exceptions.RequestException as e:
    print("An error occurred while fetching the page:", e)
