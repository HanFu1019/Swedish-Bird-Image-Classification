# Swedish-Bird-Image-Classification

This project utilizes deep learning techniques to automate bird species identification, specifically focusing on species found in Sweden. By leveraging transfer learning on pre-trained models like VGGNet, ResNet, and DenseNet, we achieve high accuracy in classifying 507 bird species.

## Table of Contents
- [Introduction](#introduction)
- [Scraper](#scraper)
- [Dataset](#dataset)
- [Models and Methods](#models-and-methods)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
Bird species identification is critical for fields such as conservation biology, ecology, and biodiversity research. Traditional identification methods are often labor-intensive and prone to human error. This project automates the identification process using Convolutional Neural Networks (CNNs), which can handle large image datasets with high precision. High-quality datasets from eBird and YOLOv8 for image standardization provide a strong foundation for robust, scalable bird identification models.

## Scraper
This project contains a series of scripts located in the [`scraper`](scraper) folder designed to scrape bird images from a specified location. To use the scraper, follow these steps in order:

1. **Get Bird List**: Run [`1_birds_list.py`](scraper/1_birds_list.py) to fetch a list of bird species for a specified country. To find the location code to use, visit [eBird Explore](https://ebird.org/explore) and enter your region in the "Enter a region" box. The species will be saved in `ebird_species.txt`, which contains the identifiers needed for the following steps.

2. **Download CSV Links**: Execute [`2_csv_download.py`](scraper/2_csv_download.py) to generate CSV files containing image links for each species. Note that you'll need to create an account on eBird and use your cookies (headers) in the script for authentication. You'll be prompted to enter the number of images to download per species. The CSV files will be stored in a `Dataset` directory, organized by species.

3. **Rename Folders**: Run [`3_folder_name_change.py`](scraper/3_folder_name_change.py) to rename the species folders based on the scientific names extracted from the CSV files. This step ensures the folder names reflect the correct scientific classifications.

4. **Download Images**: Finally, run [`4_images_download.py`](scraper/4_images_download.py) to download the actual images using the links from the CSV files. You can choose your desired resolution from available options, and the images will be saved in their respective species folders.


## Dataset
The dataset contains 507,000 images of Swedish bird species, sourced from eBird. Images are filtered by quality and standardized to a consistent square aspect ratio using YOLOv8, a state-of-the-art object detection model. The dataset is organized in class-labeled folders, ready for model training.

## Models and Methods
We explored and fine-tuned three CNN architectures for bird classification:
- **VGGNet**: Known for its simple and deep architecture, effective in feature extraction.
- **ResNet**: Uses residual connections to mitigate the vanishing gradient problem, allowing deeper layers.
- **DenseNet**: Enhances feature reuse by connecting each layer to every other layer, improving gradient flow.

An **ensemble model** combining ResNet and DenseNet was also implemented to maximize classification performance. All models are fine-tuned with PyTorch using transfer learning from ImageNet pre-trained weights.

## Results
The DenseNet169 model achieved the highest performance on the test dataset, with:
- **Top-1 Accuracy**: 91.1%
- **Top-5 Accuracy**: 98.2%

The results indicate that DenseNet169 effectively balances training accuracy and generalization, making it the preferred model.

## Conclusion
This project demonstrates the potential of deep learning in automating bird species identification with high accuracy. Future directions could include real-time deployment, advanced data augmentations, and exploration of novel architectures such as Vision Transformers.
