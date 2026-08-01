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

  class_names = sorted([item.name for item in os.scandir(directory)])

  if not class_names:
    raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
  class_to_idx = {name : idx for idx, name in enumerate(class_names)}

  return class_names,class_to_idx



class customImageFolder(Dataset):
  # definitely need to overwrite __getitem__, can __len__ also
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

def create_datasets(dir_name: Path, transformations:None) -> Tuple[torch.utils.data.DataLoader,torch.utils.data.DataLoader,list[str],dict[str,int]]:

  dataset = customImageFolder(target_directory=dir_name,
                    transformations= transformations)

def create_dataloaders(train_dir: Path, 
                       test_dir: Path,
                       transformations: transforms.Compose = None, 
                       NUM_WORKERS: int = os.cpu_count(),
                       BATCH_SIZE: int = 32):

  train_dataset = customImageFolder(target_directory=train_dir,
                    transformations= transformations)
  
  test_dataset = customImageFolder(target_directory=test_dir,
                    transformations= transformations)
  
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


