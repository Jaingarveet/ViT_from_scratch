# still need to annotate functions
# still need to add type checks!
from typing import Tuple, List, Dict
from PIL import Image
import torch
import zipfile
from pathlib import Path
import requests
import os
import random

def get_data():

  root_path = Path(__file__).parent.resolve()
  data_path = root_path / "data"
  image_path = data_path / "pizza_steak_sushi"

  if image_path.is_dir():
      print(f"{image_path} directory exists.")
  else:
      print(f"Did not find {image_path} directory, creating one...")
      image_path.mkdir(parents=True, exist_ok=True)

      # Download pizza, steak, sushi data
      with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
          request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
          print("Downloading pizza, steak, sushi data...")
          f.write(request.content)

      # Unzip pizza, steak, sushi data
      with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
          print("Unzipping pizza, steak, sushi data...")
          zip_ref.extractall(image_path)

def walk_through_dir(dir_path):
  for root, dirs, files in os.walk(dir_path):
    print(f"there are {len(dirs)} directories and {len(files)} images in inside {root} directory")

def plot_images(image_paths, transform = None, n=3, seed= 42):
  # customized for plotting transformed image or not 
  random.seed(42)
  random_samples_path = random.sample(image_paths,k=n)
  for image_path in random_samples_path:
      fig , ax = plt.subplots(1,2)
      image_class = image_path.parent.stem
      image = Image.open(image_path)
      transformed_image = transform(image).permute(1,2,0)

      fig.suptitle(f"Class: {image_class}")

      ax[0].imshow(image)
      ax[0].set_title('original image')
      ax[0].axis(False)

      ax[1].imshow(transformed_image)
      ax[1].set_title('transformed_image')
      ax[1].axis(False)

def get_device() -> str:

  # should I add the mac device thing as well, also would I need to change when I add accelerator?
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return device


def find_classes(directory: str)-> Tuple[List[str],Dict[str,int]]:

  class_names = sorted([item.name for item in os.scandir(directory)])

  if not class_names:
    raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
  class_to_idx = {name : idx for idx, name in enumerate(class_names)}

  return class_names,class_to_idx


def display_random_image(
    dataset: torch.utils.data.Dataset,
    n: int = 10,
    display_shape: bool = True,
    seed: int = None):

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




def plot_loss_curves(results:dict) -> None:

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

# The 2 functions below (pred_and_plot_image and save_model are from learnpytorch.io course from where I learned this stuff and I use that code normally for my boilerplate.)

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image and plots the image with its prediction."""

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.

    # 3. Transform if necessary
    if transform:
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
    plt.axis(False);




def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
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
