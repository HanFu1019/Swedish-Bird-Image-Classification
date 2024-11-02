import os
import pandas as pd
import aiohttp
import asyncio
import time
import random
from tqdm import tqdm
from aiohttp import ClientSession

# Supported resolutions
SUPPORTED_RESOLUTIONS = [160, 320, 480, 640, 900, 1200, 1800, 2400]

# Function to prompt for resolution selection
def choose_resolution():
    print("Available resolutions: ", ", ".join(map(str, SUPPORTED_RESOLUTIONS)))
    while True:
        try:
            resolution = int(input("Please enter the desired resolution: "))
            if resolution in SUPPORTED_RESOLUTIONS:
                return resolution
            else:
                print(f"Invalid choice. Please select one of: {SUPPORTED_RESOLUTIONS}")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Function to download an image with retries and error handling
async def download_image(semaphore, session, image_url, save_path, retries=3):
    """Download an image with retries, handling rate limits and forbidden access."""
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        with open(save_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        return f"Downloaded to {save_path}"
                    elif response.status == 429:  # Rate limited
                        retry_after = response.headers.get('Retry-After', None)
                        delay = float(retry_after) if retry_after else (2 ** attempt) + random.uniform(2, 10)
                        print(f"Rate limited. Retrying in {delay:.2f} seconds...")
                        await asyncio.sleep(delay)
                    elif response.status == 403:  # Forbidden (likely blocked)
                        print("403 Forbidden detected. Manual IP change required.")
                        return "Access forbidden. Manual IP change required."
                    else:
                        print(f"Failed with status {response.status}. Retrying...")
            except Exception as e:
                print(f"Exception encountered: {e}. Retrying...")
            await asyncio.sleep(random.uniform(1, 3))  # Random sleep to avoid detection
    return f"Failed to download {image_url} after {retries} attempts."

# Asynchronous function to download images with concurrency control
async def download_images(df, folder_path, resolution, max_concurrency):
    semaphore = asyncio.Semaphore(max_concurrency)
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/16.16299',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 13_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Safari/605.1.15',
        'Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.72 Mobile Safari/537.36',
    ]
    headers = {'User-Agent': random.choice(user_agents)}

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for _, row in df.iterrows():
            image_url = f"https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{row['ML Catalog Number']}/{resolution}"
            save_path = os.path.join(folder_path, f"{row['ML Catalog Number']}.jpg")
            tasks.append(download_image(semaphore, session, image_url, save_path))
        
        results = await asyncio.gather(*tasks)
        return results

# Main script
def main():
    dataset_dir = 'Dataset'
    resolution = choose_resolution()
    max_concurrency = 2  # Adjust as needed to avoid rate limits

    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        if os.path.isdir(folder_path):
            csv_file = next((f for f in os.listdir(folder_path) if f.endswith('.csv')), None)
            if csv_file:
                csv_path = os.path.join(folder_path, csv_file)
                df = pd.read_csv(csv_path)

                while True:
                    try:
                        results = asyncio.run(download_images(df, folder_path, resolution, max_concurrency))
                        if any("Access forbidden" in result for result in results):
                            print("Access forbidden detected. Please change IP and press Enter to continue.")
                            input("Press Enter after changing IP...")
                            continue
                        
                        # Message when download completes for a CSV
                        print(f"Completed downloading images for {csv_file} in folder '{folder_name}'")
                        break
                    except Exception as e:
                        print(f"Error encountered: {e}. Retrying after IP change...")
                        input("Press Enter after changing IP to retry...")
                        continue

# Run the main script
if __name__ == "__main__":
    main()
