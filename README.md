# Swedish-Bird-Image-Classification

This project utilizes deep learning techniques to automate bird species identification, specifically focusing on species found in Sweden. By leveraging transfer learning on pre-trained models like VGGNet, ResNet, and DenseNet, we achieve high accuracy in classifying 507 bird species.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models and Methods](#models-and-methods)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
Bird species identification is critical for fields such as conservation biology, ecology, and biodiversity research. Traditional identification methods are often labor-intensive and prone to human error. This project automates the identification process using Convolutional Neural Networks (CNNs), which can handle large image datasets with high precision. High-quality datasets from eBird and YOLOv8 for image standardization provide a strong foundation for robust, scalable bird identification models.

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
