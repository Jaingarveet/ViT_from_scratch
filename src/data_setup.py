from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, List, Dict
import torch
from PIL import Image
import zipfile
from pathlib import Path
import requests
import os
import matplotlib.pyplot as plt
from pathlib import Path

def get_data():
  """
  This function downloads and extracts the dataset we currently use inside a separate directory.
  Skips downloading if the data already exists.  
  """
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


def find_classes(directory: str)-> Tuple[List[str],Dict[str,int]]:
  """
  This function accepts a directory path which is supposed to contain all the images from dataset in the form of separate sub-directories 
  where each sub-directory's name is the label of that image which helps us find the actual class names.
  Input: 
  directory: directory path containing individual sub-directory for each class from dataset
  Output: a tuple containg the actual class_names as the first argument and a dictionary mapping from class to index as second argument
  """

  class_names = sorted([item.name for item in os.scandir(directory)])

  if not class_names:
    raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
  class_to_idx = {name : idx for idx, name in enumerate(class_names)}

  return class_names,class_to_idx



class customImageFolder(Dataset):
  # definitely need to overwrite __getitem__, can __len__ also
  """ 
  Custom Dataset implementation to load image from the data directory into a useful pytorch dataset using 
  torch.utils.data.Dataset. Accepts the target directory containing the data and the transformations to be used on that data as 
  arguments.
  """
  def __init__(self,target_directory:str ,transformations = None) -> None:

    self.paths = list(Path(target_directory).glob('*/*.jpg'))

    self.class_names, self.class_to_idx = find_classes(target_directory)

    self.transformations = transformations


  def load_image(self, index: int) -> Image.Image:
     image_path = self.paths[index]
     return Image.open(image_path)

  def __len__(self) ->int:
    return len(self.paths)

  def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
      image = self.load_image(index)
      label = self.paths[index].parent.name
      class_idx = self.class_to_idx[label]
      if self.transformations:
            return self.transformations(image), class_idx
      else:
        return image, class_idx

def create_datasets(train_dir: Path, 
                    test_dir: Path,
                    train_transformations: transforms.Compose = None, 
                    test_transformations: transforms.Compose = None, 
                    NUM_WORKERS: int = os.cpu_count(),
                    BATCH_SIZE: int = 32) -> Tuple[torch.utils.data.DataLoader,torch.utils.data.DataLoader,list[str],dict[str,int]]:
  """
  Utilizes the customImageFolder class to create custom datasets for training and testing and respectively creates training and testing dataloaders.
  INPUT: 
  train_dir: Path to directory containing training images
  test_dir: Path to directory containing testing images
  train_transformations: transformations to be used on the images to train the model
  test_transformations: transformations to be used on the images to test the model
  pre-process dataset, DEFAULT=None 
  NUM_WORKERS: Number of workers to be used for loading data while using dataloader, DEFAULT=os.cpu_count()
  BATCH_SIZE: Batch size for creating dataloaders, DEFAULT=32

  RETURNS:
  train_dataloader: torch.utils.data.DataLoader
  test_dataloader: torch.utils.data.DataLoader
  class_names: list containg the actual class_names
  class_to_idx: dictionary mapping from class to index
  """

  train_dataset = customImageFolder(target_directory=train_dir,
                    transformations= train_transformations)
  # separate transformations for training and testing in order to separate cases of data augmentation and similar things
  test_dataset = customImageFolder(target_directory=test_dir,
                    transformations= test_transformations)
  
  class_names = train_dataset.class_names
  class_to_idx = train_dataset.class_to_idx

  train_dataloader = DataLoader(dataset = train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=NUM_WORKERS)

  test_dataloader = DataLoader(dataset = test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS)

  return train_dataloader, test_dataloader, class_names, class_to_idx


