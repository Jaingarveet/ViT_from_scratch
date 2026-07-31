# VLM_pipeline

This repo contains my revision of pytorch, I will try building a VLM from scratch if time allows me to, and I hopefully do end up doing that... I try to read documentation along with coding so there might be a point I might end up committing notes and code both as well or just some pointers for me.

To be continued.....

Plan would be to revise pytorch first -> implement minimal working VLM -> optimize it with additional things and
production ready structure
right now: add UV initialization, install dependencies, add gitignore
We will see, max 1 month

BASELINE?
collation needed or no?


## Hopeful scope over next month
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
* Things from the lectures and ITDS for custom loss functions
* Course related optimizations present in extra curriculars and stuff, like operator fusion
* object detection pipeline?
* https://pytorch.org/blog/tensor-memory-format-matters/#pytorch-best-practice
