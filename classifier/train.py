import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import torch
import seaborn as sns

# pytorch imports
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def train_epoch(model, data_loader, loss_fn,
                optimizer, device, scheduler, n_examples ):

  # switch model to training mode
  model = model.train()

  # store losses for each iteration
  losses = []
  correct_predictions = 0

  # loop through the dataset batches
  for batch_id, d in enumerate(data_loader):
    if batch_id % 1000 == 0:
      print("Training batch : {}/{}".format(batch_id, len(data_loader)))
    # store in gpu
    input_ids = d["encoding"].to(device)
    attention_mask = d["attention_mask"].to(device)
    labels = d["label"].to(device)

    # perform model predictions
    outputs = model(
      encodings=input_ids,
      attention_mask=attention_mask
    )

    # the prediction with max value represents the class
    _, preds = torch.max(outputs, dim=1)

    correct_predictions += torch.sum(preds == labels)

    # compute loss from predictions
    loss = loss_fn(outputs, labels)  
    losses.append(loss.item())
    loss.backward()

    # clipping the gradients to prevent exploding grads
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # update model weights using optimizer
    optimizer.step()

    scheduler.step()

    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  # switch model to evaluation mode
  model = model.eval()

  # store losses for each iteration
  losses = []
  correct_predictions = 0

  # prevents pytorch from storing grads
  # for evaluation we do not update weights
  # thus we don't require grads
  with torch.no_grad():

    # loop through the dataset batches
    for batch_id, d in enumerate(data_loader):
      if batch_id % 1000 == 0:
        print("Evaluation batch : {}/{}".format(batch_id, len(data_loader)))

      # store values in gpu
      input_ids = d["encoding"].to(device)
      attention_mask = d["attention_mask"].to(device)
      labels = d["label"].to(device)

      # perform model predictions
      outputs = model(
        encodings=input_ids,
        attention_mask=attention_mask
      )

      # the prediction with max value represents the class
      _, preds = torch.max(outputs, dim=1)

      correct_predictions += torch.sum(preds == labels)

      # compute loss from predictions
      loss = loss_fn(outputs, labels)      
      losses.append(loss.item())
      
  return correct_predictions.double() / n_examples, np.mean(losses)