# still need to annotate functions
# still need to add type checks!
from typing import Tuple, List, Dict
from PIL import Image
import torch
from pathlib import Path
import requests
import torchvision
from torchvision import transforms
import os
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


def walk_through_dir(dir_path)-> None:
  """
  For a directory structure where the root directory contains dataset separated by their labels as name of the sub-directories,
  this function returns the quantitity of the paths in readable format.
  INPUT: dir_path: path containing label separated directories for images.
  """
  for root, dirs, files in os.walk(dir_path):
    print(f"there are {len(dirs)} directories and {len(files)} images in inside {root} directory")

def get_device() -> str:
  """
  Function to return the current available device to be used for computation.
  """
  # should I add the mac device thing as well, also would I need to change when I add accelerator?
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return device


def plot_random_images_from_path(image_paths:Path, transform : transforms.Compose = None, n=3, seed= 42)-> None:
  # customized for plotting transformed image or not 
  """
  Plots random images from given path list, try to limit the value of random samples to maximum of 3
  INPUT: 
  image_paths: list of paths of images from the working directory of model
  transform: transformations to be use on the image
  n: number of images to display (maximum of 3 and default = 3)
  seed: random seed, DEFAULT = 42
  """
  if transform is None:
     transform = transforms.Compose([transforms.ToTensor()])
     
  random.seed(42)
  random_samples_path = random.sample(image_paths,k=n)
  for image_path in random_samples_path:
      fig , ax = plt.subplots(1,2)
      image_class = image_path.parent.stem
      image = Image.open(image_path)
      transformed_image = transform(image).permute(1,2,0)

          
      fig_ttl = fig.suptitle(f"Class: {image_class}")
      fig_ttl.set_position([.5, 0.8])

      ax[0].imshow(image)
      ax[0].set_title('original image')
      ax[0].axis(False)

      ax[1].imshow(transformed_image)
      ax[1].set_title('transformed_image')
      ax[1].axis(False)


def display_random_images_from_dataset(
    dataset: torch.utils.data.Dataset,
    n: int = 10,
    display_shape: bool = True,
    seed: int = None) -> None:
  
  """
    Plots random images from across the dataset, try to limit the value of random samples to maximum of 10
    INPUT: 
    dataset: torch.utils.data.Dataset to display images from
    n: number of images to display (maximum of 3 and default = 3)
    display_shape: Boolean to whether to display the image shape along with the plot or not
    seed: random seed, DEFAULT = 42
  """

  if n>10:
    n=10
    print("Can't have n > 10, setting n to 10")
  if seed:
    random.seed(seed)

  random_samples_idx = random.sample(range(len(dataset)), k=n)
  fix, ax = plt.subplots(len(random_samples_idx),1)
  classes = dataset.class_names

  for i, random_idx in enumerate(random_samples_idx):
    image, class_label = dataset[random_idx][0], dataset[random_idx][1]
    image_adjusted = image.permute(1,2,0)

    ax[i].imshow(image_adjusted)
    ax[i].set_title(classes[class_label])
    ax[i].axis(False)


def plot_loss_acc_curves(results:dict) -> None:
    """
      Plots the loss and accuracy values obtained from training and testing the model
      INPUT: 
      results: dictionary containing keys as train_loss, train_acc, test_loss, test_acc and their corresponding values over epochs as values
    """

    train_loss = results['train_loss']
    train_acc = results['train_acc']
    test_loss = results['test_loss']
    test_acc = results['test_acc']

    epochs = range(len(train_loss))

    plt.figure(figsize=(13,8))

    plt.subplot(1,2,1)
    plt.plot(epochs,train_loss,label='train_loss')
    plt.plot(epochs,test_loss,label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs,train_acc,label='train_acc')
    plt.plot(epochs,test_acc,label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = 'cpu')->None:

    """Make a prediction on target image and show the image with the predicted label and it's predicted probability.
    INPUT:
    model: Model to use for prediction
    image_path: image path to be read and predict the label and probability for
    class_names: a list of class names that are predicted on, DEFAULT=None
    transform: transformations to be used to pre-process the image to be used by the model,DEFAULT=None
    device: device on which computation to perform on, DEFAULT=cpu
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.

    # 3. Transform if necessary
    if transform is not None:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seed(seed:int=42):
   """
   sets seed across CPU and GPU operations and for GPU accelration as well.
   input: seed number
   """
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)

def load_model(model: torch.nn.Module,
               target_file: str) -> torch.nn.Module:
  """
  Loads a pytorch model from a target file
  INPUT:
  pass a class of the model,
  pass the target_file where state_dict is stored,
  load the model and return it 
  """
  assert target_file.endswith(".pth") or target_file.endswith(".pt"), "model_name should end with '.pt' or '.pth' when stored"
  model.load_state_dict(torch.load(target_file))
  return model


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str) -> None:
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

def print_patched_image(image:torch.Tensor, label:str, patch_resolution:int=16):
    fig, ax = plt.subplots(14, 14, figsize=(5,5))
    permute_image = image.squeeze(dim=0).permute(1,2,0)
    for i in range(14):
      for j in range(14):
        ax[i,j].imshow(permute_image[(i * patch_resolution):((i+1)*patch_resolution),
                                  (j * patch_resolution):((j+1)*patch_resolution),
                                  :])
        ax[i,j].axis(False)
        plt.suptitle(f"patched version of the image labeled: {label}")

def plot_confusion_matrix(model:torch.nn.Module, dataloader: torch.utils.data, device: str,class_names:list[str]):
        
      y_pred = []
      y_true = []

      # iterate over test data
      for inputs, labels in tqdm(dataloader):
              model.to(device), inputs.to(device), labels.to(device)
              output = model(inputs) # Feed Network

              pred_logits = torch.softmax(output,dim=1)
              pred_labels = torch.argmax(pred_logits,dim=1).data.cpu().numpy()
              y_pred.extend(pred_labels) # Save Prediction

              labels = labels.data.cpu().numpy()
              y_true.extend(labels) # Save Truth

      classes = class_names

      cf_matrix = confusion_matrix(y_true, y_pred)
      df_cm = pd.DataFrame(cf_matrix , index = [i for i in classes],
                          columns = [i for i in classes])
      plt.figure(figsize = (12,7))
      ax = sn.heatmap(df_cm, annot=True, cmap='Set2')
      ax.set(xlabel='Predicted labels',ylabel='True labels')
      plt.show() 

# def profile_vit
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#              record_shapes=True,
#              schedule=torch.profiler.schedule(wait=1, warmup=1, active=2)
#              ) as vit_train_profiler:
#   with record_function("vit_train__optimizing_debug"):
#            for _ in range(4):
#             vit_train_results = train(model = ViT_model,
#                 train_dataloader= vit_train_dataloader,
#                 test_dataloader= vit_test_dataloader,
#                 optimizer= vit_optimizer,
#                 device = device,
#                 loss_fn= vit_loss_fn,
#                 epochs=NUM_EPOCHS
#                 )
#             vit_train_profiler.step()