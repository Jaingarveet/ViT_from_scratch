# VLM_pipeline

This repo contains my revision of pytorch, I will try building a VLM from scratch if time allows me to, and I hopefully do end up doing that... I try to read documentation along with coding so there might be a point I might end up committing notes and code both as well or just some pointers for me.

#### Image captioning datasets it is...

To be continued.....

Plan would be to revise pytorch first -> implement minimal working VLM -> optimize it with additional things and
production ready structure
right now: add UV initialization, install dependencies, add gitignore
what benchmarks to add?

BASELINE: I first implement the model using PyTorch primitives to establish a reliable baseline. After validating the pipeline, I progressively replaced key components such as Multi-Head Attention and Transformer blocks with my own implementations, verifying that performance remained consistent through ablation testing.

Investigate robustness of image-captioning pipelines under controlled visual perturbations and distribution shifts.

# Hopeful scope till october end: 
https://chatgpt.com/s/t_6a74882a84f081918e6a430092c82fd9
implement the baseline image captioning model anyways to gradually progress it into object detection in VLM.




collation needed or no?
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
- - add a logger throughout the modular code later for deployment and monitoring purposes and check if MLFLOW provides a method to deploy models or not? and CI/CD
**implement the loss function yourself. and manual datasets, like manual things as much as possible but limited overall scope**
argparsing based workflow or direct main.ipynb and rest in python script?
## Hopeful scope over next 3 months:
* No need for custom dataset, get directly from hugging face, instead we will use a custom tokenizer
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

