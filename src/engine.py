import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Tuple

def train_step(model:torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float,float]:

  model.train()
  model.to(device)
  train_loss = 0
  train_acc = 0

  for i, (X,y) in enumerate(dataloader):
    X.to(device), y.to(device)
    pred_logits = model(X)
    loss = loss_fn(pred_logits,y)
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred_label = torch.argmax(torch.softmax(pred_logits,dim=1),dim=1)
    train_acc += (pred_label == y).sum().item() / len(pred_logits)
    # this is accuracy on this batch

    print(f"Completed {i} batch out of {len(dataloader)} for training")

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  # average batch accuracy

  return train_loss, train_acc


def test_step(model:torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device) -> Tuple[float,float]:

  model.eval()
  model.to(device)
  test_loss = 0
  test_acc = 0
  with torch.inference_mode():
    for i, (X,y) in enumerate(dataloader):
      X.to(device), y.to(device)
      pred_logits = model(X)
      loss = loss_fn(pred_logits,y)
      test_loss += loss.item()

      pred_label = torch.argmax(torch.softmax(pred_logits,dim=1),dim=1)
      test_acc += (pred_label == y).sum().item() / len(pred_logits)
      # this is accuracy on this batch

      print(f"Completed {i} batch out of {len(dataloader)} for testing")

    test_loss /= len(dataloader)
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

  results = {'train_loss':[],
             'train_acc':[],
             'test_loss':[],
             'test_acc':[],
             }
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
