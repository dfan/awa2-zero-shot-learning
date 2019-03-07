import argparse
import numpy as np
from AnimalDataset import AnimalDataset
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import os
import sys

def build_model(num_labels, is_parallel):
  model = torchvision.models.resnet50().to(device)
  if is_parallel:
    print('Using DataParallel:')
    model = nn.DataParallel(model)
    model_features = model.module.fc.in_features
    model.module.fc = nn.Sequential(nn.BatchNorm1d(model_features), nn.Dropout(p=0.5), nn.Linear(model_features, num_labels))
  else:
    print('Not using DataParallel:')
    model_features = model.fc.in_features
    model.fc = nn.Sequential(nn.BatchNorm1d(model_features), nn.Dropout(p=0.5), nn.Linear(model_features, num_labels))
  return model

def train(num_epochs, eval_interval, learning_rate, model_name, optimizer_name, batch_size):
  train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 3}
  test_params = {'batch_size': 1, 'shuffle': True, 'num_workers': 3}
  train_process_steps = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.Resize((224,224)), # ImageNet standard
    transforms.ToTensor()
  ])
  test_process_steps = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
  ])
  train_dataset = AnimalDataset('trainclasses.txt', train_process_steps)
  test_dataset = AnimalDataset('testclasses.txt', test_process_steps)
  train_loader = data.DataLoader(train_dataset, **train_params)
  test_loader = data.DataLoader(test_dataset, **test_params)
  criterion = nn.BCELoss()

  total_steps = len(train_loader)
  if torch.cuda.device_count() > 1:
    model = build_model(num_labels, True).to(device)
  else:
    model = build_model(num_labels, False).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  for epoch in range(num_epochs):
    for i, (images, features, img_names, indexes) in enumerate(train_loader):
      # Batchnorm1D can't handle batch size of 1
      if images.shape[0] < 2:
        break
      images = images.to(device)
      features = features.to(device).float()
      # Toggle training flag
      model.train()

      outputs = model(images)
      sigmoid_outputs = torch.sigmoid(outputs)
      loss = criterion(sigmoid_outputs, features)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if i % 50 == 0:
        curr_iter = epoch * len(train_loader) + i
        print ('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
        sys.stdout.flush()

    # Do some evaluations
    if (epoch + 1) % eval_interval == 0:
      print('Evaluating:')
      curr_acc = evaluate(model, test_loader)
      print('Epoch [{}/{}] Approx. training accuracy: {}'.format(epoch+1, num_epochs, curr_acc))
  
  # Make final predictions
  print('Making predictions:')
  if not os.path.exists('models'):
    os.mkdir('models')
  torch.save(model.state_dict(), 'models/{}'.format(model_name))
  torch.save(optimizer.state_dict(), 'models/{}'.format(optimizer_name))
  make_predictions(model, test_loader)

def labels_to_class(pred_labels):
  predictions = []
  for i in range(pred_labels.shape[0]):
    curr_labels = pred_labels[i,:].cpu().detach().numpy()
    min_dist = sys.maxsize
    min_index = -1
    for j in range(predicate_binary_mat.shape[0]):
      class_labels = predicate_binary_mat[j,:]
      hamming_dist = np.sum(curr_labels != class_labels)
      if hamming_dist < min_dist:
        min_index = j
        min_dist = hamming_dist
    predictions.append(classes[min_index])
  return predictions

def evaluate(model, dataloader):
  # Toggle flag
  model.eval()
  mean_acc = 0.0

  pred_classes = []
  truth_classes = []
  with torch.no_grad():
    for i, (images, features, img_names, indexes) in enumerate(dataloader):
      images = images.to(device)
      features = features.to(device).float()
      outputs = model(images)
      sigmoid_outputs = torch.sigmoid(outputs)
      pred_labels = sigmoid_outputs > 0.5
      curr_pred_classes = labels_to_class(pred_labels)
      pred_classes.extend(curr_pred_classes)

      curr_truth_classes = []
      for index in indexes:
        curr_truth_classes.append(classes[index])
      truth_classes.extend(curr_truth_classes)
  
  pred_classes = np.array(pred_classes)
  truth_classes = np.array(truth_classes)
  mean_acc = np.mean(pred_classes == truth_classes)

  # Reset
  model.train()
  return mean_acc

def make_predictions(model, dataloader):
  # Toggle flag
  model.eval()
  
  pred_classes = []
  output_img_names = []
  with torch.no_grad():
    for i, (images, features, img_names, indexes) in enumerate(dataloader):
      images = images.to(device)
      features = features.to(device).float()
      outputs = model(images)
      sigmoid_outputs = torch.sigmoid(outputs)
      pred_labels = sigmoid_outputs > 0.5
      curr_pred_classes = labels_to_class(pred_labels)
      pred_classes.extend(curr_pred_classes)
      output_img_names.extend(img_names)
      
      if i % 1000 == 0:
        print('Prediction iter: {}'.format(i))

    with open('predictions.txt', 'w') as f:
      for i in range(len(pred_classes)):
        output_name = output_img_names[i].replace('data/JPEGImages/', '')
        f.write(output_name + ' ' + pred_classes[i] + '\n')
      
def load_model(model_file):
  is_parallel = True # torch.cuda.device_count() > 1
  model = build_model(num_labels, is_parallel).to(device)
  if is_parallel:
    model = torch.nn.DataParallel(model)
    dict = torch.load(model_file)
    model = model.module
    model.load_state_dict(dict)
  else:
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
  return model

def debug(model_file, mode):
  model = load_model(model_file)
  test_params = {'batch_size': 1, 'shuffle': True, 'num_workers': 3}
  process_steps = transforms.Compose([
    #transforms.RandomRotation(15),
    #transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.3),
    transforms.Resize((224,224)), # ImageNet standard
    transforms.ToTensor()
  ])
  test_dataset = AnimalDataset('testclasses.txt', process_steps)
  test_loader = data.DataLoader(test_dataset, **test_params)
  if mode == 'evaluate':
    print(evaluate(model, test_loader))
  elif mode == 'predict':
    make_predictions(model, test_loader)

# Sample usage: `python train.py -n 25 -et 5 -lr 0.000025 -bs 24`
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', '-n', type=int, default=25)
  parser.add_argument('--eval_interval', '-et', type=int, default=5)
  parser.add_argument('--learning_rate', '-lr', type=float, default=0.00001)
  parser.add_argument('--model_name', '-mn', type=str, default='model.bin')
  parser.add_argument('--optimizer_name', '-opt', type=str, default='optimizer.bin')
  parser.add_argument('--batch_size', '-bs', type=int, default=24)
  args = parser.parse_args()
  args = vars(args)

  num_epochs = args['num_epochs']
  eval_interval = args['eval_interval']
  learning_rate = args['learning_rate']
  model_name = args['model_name']
  optimizer_name = args['optimizer_name']
  batch_size = args['batch_size']

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  classes = np.array(np.genfromtxt('data/classes.txt', dtype='str'))[:,-1]
  predicates = np.array(np.genfromtxt('data/predicates.txt', dtype='str'))[:,-1]
  predicate_binary_mat = np.array(np.genfromtxt('data/predicate-matrix-binary.txt', dtype='int'))
  predicate_continuous_mat = np.array(np.genfromtxt('data/predicate-matrix-continuous.txt', dtype='float'))
  num_labels = len(predicates)
  train(num_epochs, eval_interval, learning_rate, model_name, optimizer_name, batch_size)

  #debug('models/model.bin', 'predict')
