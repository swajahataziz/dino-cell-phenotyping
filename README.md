# Distributed Training of Self-Supervised Vision Transformers in PyTorch for multi-channel cell images with Amazon SageMaker

Vision Transformers are increasingly popular in the computational drug discovery space, especially in the area of phenotypic characterisation using multi-channel cell images. In this project, we will show how to pretrain and do downstream analyses for [Meta's DINO](https://github.com/facebookresearch/dino/tree/master) model.

### Download Deep Phenotyping Image Data
Download the  [Filamentous Fungi using Cell Painting dataset](https://zenodo.org/records/8227399) and upload it to an S3 bucket. The dataset needs to be restructured in PyTorch format where images pertaining to each class are under their respected subfolders. 
For more information about the dataset, please refer to the [paper](https://www.biorxiv.org/content/10.1101/2023.08.24.554566v1.full)

## 2. Run Distributed Training jobs
The notebook 'dinov1-training.ipynb' demonstrates how to set up distributed training in SageMaker using S3 as well as FSx as the data repository. 
