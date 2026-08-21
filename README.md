# VLM_pipeline

This repo contains my case study of pytorch, I will try building a VLM from scratch if time allows me to, and I hopefully do end up doing that... I try to read documentation along with coding so there might be a point I might end up committing notes and code both as well or just some pointers for me.

#### Image captioning datasets it is...

To be continued.....

Plan would be to revise pytorch first -> implement minimal working VLM -> optimize it with additional things and
production ready structure
right now: add UV initialization, install dependencies, add gitignore
what benchmarks to add?
busy wait, spinlocking and stuff.
BASELINE: I first implement the model using PyTorch primitives to establish a reliable baseline. After validating the pipeline, I progressively replaced key components such as Multi-Head Attention and Transformer blocks with my own implementations, verifying that performance remained consistent through ablation testing.

Investigate robustness of image-captioning pipelines under controlled visual perturbations and distribution shifts.

# Hopeful scope till october end: 
https://chatgpt.com/s/t_6a74882a84f081918e6a430092c82fd9
implement the baseline image captioning model anyways to gradually progress it into object detection in VLM.
- Ask Ekta vats for model knowledge, and salman toor for things like wether to use redis or something.


do the deterministic transformations beforehand before the dataloaders in training loop to prevent the overhead, check profiling for more designs
Doesn't have to be all the ones, just something that are independent?
https://chatgpt.com/s/t_6a7defa5b7988191b84bf8f7e92b58ae
collation needed or no?
DDP 

 mixed precision 

gradient accumulation

distributed checkpointing

compilation
apache pulsar for data pre-processing?
- Go through Model deployment and MLOPs stuff after the actual image captioning pipeline is working.
- Add logging at the time when we convert the captioning pipeline into a separate python script for running more experiments, along with argparser as well.
- Add ViT in actual code, experiment with adding a timer for measuring times as logs for training, along with comparing it to pre-trained model
- - Add annotations in the functions as well
- add a logger throughout the modular code
- improve the utility functions
- add a coefficient matrix and top n most wrong predictions
- add data augmentation to code as well
- improve the get_data functions to add data to a particular destination or something, source, destination, remove_source, return image_path
- use MLFLOW on top of that to track experiments, compare it to the pre-trained ViT model, save model and stuff, checkout the experiment tracking guide for best practices, vertical and horizontal scaling experiments,
- First add experiment tracking using MLFLOW obviously and add experiments to track vertical and horizontal scaling. Mention the law for vertical scaling or not?
- Can we also add experiments to use different torch data type for making the model?
- Take colab subscription to train a private model and publish it on hugging face hub using model quantization and stuff.
- check if I need to add stratification in preprocessing and lookup never to miss steps in pre-processing
- instead of ray we are gonna be using an AWS EC2 for 50 dollar credits to run multi-gpu training setup on 50000 images
- - add a logger throughout the modular code later for deployment and monitoring purposes and check if MLFLOW provides a method to deploy models or not? and CI/CD
**implement the loss function yourself. and manual datasets, like manual things as much as possible but limited overall scope**
argparsing based workflow or direct main.ipynb and rest in python script?
## Hopeful scope over next 3 months:
* No need for custom dataset, get directly from hugging face, instead we will use a custom tokenizer
* openCV for video processing
* Things to explore apart from core code logic.
* VLM
* Weights and biases or MLFLOW?
* Cloud deployment (probably a cloudfare backend) 
* Distributed training
* Accelrators (is it just device agnostic code?)
* Ray Tune
* Understanding/Revising the mathematics
* Also what is vLLM?
* Proper MLOPS pipeline
* githooks for CI/CD or MLFLOW? Like I am anyways gonna use this for my experiment tracking so...
* torchmetrics custom?
* https://docs.pytorch.org/tutorials/index.html
* try to do as scratch things as possible, custom loss functions and stuff
* Add EDA, confusion matrix, correlation(a little skeptical for generative models?)?
* torchinfo summary
* mixed precision training
* Things from the lectures and ITDS for custom loss functions
* Course related optimizations present in extra curriculars and stuff, like operator fusion
* check my brainstorm session for more, things like how to test, what kind of tests? What kind of studies?
* https://pytorch.org/blog/tensor-memory-format-matters/#pytorch-best-practice
Use dataclasses or Pydantic models for configurations.
Add type hints throughout the code.
Write a few unit tests for the dataset and model forward pass.
Use structured logging instead of lots of print() statements.
Ensure training can resume from a checkpoint without manual changes.
A nice stretch goal is to include model registry with MLflow. After each run, automatically register the best-performing model, and provide a simple inference script that loads the latest registered version and generates captions for new images. That shows you understand not just training, but the transition toward deployment. for CI/CD and add githooks for this.

- Directly use pytorch DDP for training on EC2. S3 for data loader streams
- Add Ray train/data for pre-processing and training first,
- Then add MLFLOW for experiment tracking using ray tune

- Implement an Object detection using VLM architecture from a paper

Use Ray tune, and other stuff for distributed training and stuff, also Look into S3 again, arg parser to train the model using python.
Amazon S3 for data ingestion pipeline.

-Once the baseline ViT is working, move on to reading papers about VisionLanguage Model and checkout the existing bookmarks., also see what is vLLM then?

- Try to do the ViT model from existing layers of pytorch, 
Kuberentes, cloudflare, CI/CD all the things that I could add from the web dev course.

- Do experiments to see if it is actually working on a slightly bigger dataset, like FashionMNIST or something, also change the get_data function to modify the URL and add a source destination and remove the zip file, read the attention is all you need paper, and the an image is worth 16x 16 words,
Change the existing layers to custom layers, early stopping and common regularisation technique mentioned through the paper,
Mixed precision training. Representative examples of attention for object detection using the thing mentioned in our paper.

Problem Framing & Model Selection: Knowing why tree-based ensembles (Random Forest, XGBoost, LightGBM) dominate tabular data, versus why linear/logistic models excel on sparse data or when strict interpretability is required.
Validation & Data Leakage: Recognizing why a standard random $k$-fold split breaks on time-series or grouped data, and knowing how to prevent target leakage during feature engineering.
Metric Alignment: Knowing why accuracy is useless on imbalanced datasets, and when to optimize for Precision, Recall, PR-AUC, ROC-AUC, or Log-Loss based on the business cost of false positives vs. false negatives.
Bias-Variance Diagnosis: Looking at training versus validation error curves and instantly knowing whether to add regularization, collect more data, reduce model capacity, or engineer new features.


revise the main mathematics like loss functions, negative log likelihood, expectations, matrix representations. 