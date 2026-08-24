import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Tuple
from datetime import datetime

def train_step(model:torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float,float]:
  """
    This function contains the training loop which does the forward pass through the model using data obtained from dataloader, 
    calculates the loss, does backwardpass on loss and finally optimizes the gradients.
    note: this function only implements a single iteration through complete dataset, not taking epochs into account

    INPUT: 
    model: Pytorch model or any pre-trained model that is suitable with pytorch api
    dataloader: training dataloaders for loading the data and labels
    loss_fn: loss function used for calculating the loss after forward pass
    optimizer: Optimizer to manage the optimization process of gradients
    device: Device on which computations will be done
    
    OUTPUT: 
    train_loss: average training loss for the whole dataset
    train_acc: average training accuracy for the whole dataset
  """
  model.train()
  model.to(device)
  train_loss = 0
  correct_predictions = 0
  total_predictions = 0
  train_acc = 0
  

  for i, (X,y) in enumerate(dataloader):
    X,y = X.to(device), y.to(device)
    pred_logits = model(X)
    loss = loss_fn(pred_logits,y)
    train_loss += loss.detach()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred_label = torch.argmax(pred_logits,dim=1)
    correct_predictions += (pred_label == y).sum().item()
    total_predictions += len(pred_label)
    del loss, pred_logits
    # this is accuracy on this batch

    print(f"Completed {i+1} batch out of {len(dataloader)} for training")
  if device =='cuda':
    train_loss = train_loss.item()/ len(dataloader)
  else: 
    train_loss = train_loss/ len(dataloader)
  train_acc = correct_predictions / total_predictions
  # average batch accuracy

  return train_loss, train_acc


def test_step(model:torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device) -> Tuple[float,float]:
  """
      This function contains the testing loop which does the forward pass through the model using data obtained from dataloader
      and then calculates the loss.
      note: this function only implements a single iteration through complete dataset, not taking epochs into account
  
      INPUT: 
      model: Pytorch model or any pre-trained model that is suitable with pytorch api
      dataloader: testing dataloaders for loading the data and labels
      loss_fn: loss function used for calculating the loss after forward pass
      device: Device on which computations will be done
      
      OUTPUT: 
      test_loss: average testing loss for the whole dataset
      test_acc: average testing accuracy for the whole dataset
  """
  model.eval()
  model.to(device)
  test_loss = 0
  test_acc = 0
  with torch.inference_mode():
    for i, (X,y) in enumerate(dataloader):
      X,y = X.to(device), y.to(device)
      pred_logits = model(X)
      loss = loss_fn(pred_logits,y)
      test_loss += loss.detach()

      pred_label = torch.argmax(pred_logits,dim=1)
      test_acc += (pred_label == y).sum().item() / len(pred_logits)
      # this is accuracy on this batch

      print(f"Completed {i+1} batch out of {len(dataloader)} for testing")
      del loss, pred_logits

    if device =='cuda':
        test_loss = test_loss.item()/ len(dataloader)
    else: 
        test_loss = test_loss/ len(dataloader)
    test_acc /= len(dataloader)
    # average batch accuracy

    return test_loss, test_acc

def train(model:torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               test_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               loss_fn: torch.nn.Module = nn.CrossEntropyLoss,
               epochs: int = 5):
  """
        This function integrates the training and testing loop and trains the given model for 'n' number of epochs and stores the results for individual epochs
        in a structurized dictionary for representing change of values over iterations.
    
        INPUT: 
        model: Pytorch model or any pre-trained model that is suitable with pytorch api
        train_dataloader: training dataloaders for loading the data and labels
        test_dataloader: testing dataloaders for loading the data and labels
        optimizer: Optimizer to manage the optimization process of gradients
        loss_fn: loss function used for calculating the loss after forward pass, DEFAULT=nn.CrossEntropyLoss
        device: Device on which computations will be done
        epochs: Number of epochs to traing the model for, DEFAULT=5
        
        OUTPUT: 
        results: A dictionary that contains train_loss, train_acc, test_loss, test_acc as keys and their corresponding values over different epochs as values
  """

  results = {'train_loss':[],
             'train_acc':[],
             'test_loss':[],
             'test_acc':[],
             }
  start_time = datetime.now()
  for epoch in tqdm(range(epochs)):

      train_loss, train_acc = train_step(model = model,
                  dataloader = train_dataloader,
                  loss_fn = loss_fn,
                  optimizer = optimizer,
                  device=device)

      test_loss, test_acc = test_step(model = model,
                  dataloader = test_dataloader,
                  loss_fn = loss_fn,
                  device=device)

      print(
          f'Epoch: {epoch+1} |',
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
      results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
      results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
      results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

  return results