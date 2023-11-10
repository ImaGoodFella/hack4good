import torch

from tqdm import tqdm
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from torchmetrics.classification import F1Score, Precision, Recall, Accuracy


args = {'task': 'binary'}
        
metrics = nn.ModuleDict({
    "F1Score": F1Score(**args),
    "Accuracy": Accuracy(**args),
    "Precision": Precision(**args),
    "Recall": Recall(**args),
})

def train_step(model, data_loader, criterion, optimizer, device, epoch=0, print_metrics=True):

    metrics.to(device)

    model.train()
    
    tot_loss = 0
    pred_labels, true_labels = [], []
    
    for i, (img, ts_features, label) in enumerate(tqdm(data_loader)):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        
        output = model(img, ts_features)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        if print_metrics:
            pred = torch.argmax(output, dim=1)
            pred_labels.append(pred.detach().cpu().numpy())
            true_labels.append(label.detach().cpu().numpy())
        
        tot_loss += loss
    
    tot_loss /= len(data_loader)

    if print_metrics:
        all_pred = np.concatenate(pred_labels)
        all_true = np.concatenate(true_labels)
        
        f1 = f1_score(all_true, all_pred, average='macro')
        accuracy = balanced_accuracy_score(all_true, all_pred)
        precision = precision_score(all_true, all_pred, average='macro')
        recall = recall_score(all_true, all_pred, average='macro')

        print(f'Training results for epoch {epoch}')
        print('{:>8} | {:>8} | {:>9} | {:>6} | {:>5}'.format("f1-score", "accuracy", "precision", "recall", "loss"))
        print('{:>8.4f} | {:>8.4f} | {:>9.4f} | {:>6.4f} | {:>5.4f}'.format(f1, accuracy, precision, recall, tot_loss.detach().cpu().numpy()))

        res = [metric(torch.tensor(all_pred), torch.tensor(all_true)) for metric in metrics.values()]
        f1, accuracy, precision, recall = res        

        print(f'Training results for epoch {epoch}')
        print('{:>8} | {:>8} | {:>9} | {:>6} | {:>5}'.format("f1-score", "accuracy", "precision", "recall", "loss"))
        print('{:>8.4f} | {:>8.4f} | {:>9.4f} | {:>6.4f} | {:>5.4f}'.format(f1, accuracy, precision, recall, tot_loss.detach().cpu().numpy()))
    
    return tot_loss

def train_pure_step(model, data_loader, criterion, optimizer, device, epoch=0, print_metrics=True):

    model.train()
    
    tot_loss = 0
    pred_labels, true_labels = [], []
    
    for i, (img, label) in enumerate(tqdm(data_loader)):
        img, label = img.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        if print_metrics:
            pred = torch.argmax(output, dim=1)
            pred_labels.append(pred.detach().cpu().numpy())
            true_labels.append(label.detach().cpu().numpy())
        
        tot_loss += loss
    
    tot_loss /= len(data_loader)

    if print_metrics:
        all_pred = np.concatenate(pred_labels)
        all_true = np.concatenate(true_labels)
        
        f1 = f1_score(all_true, all_pred, average='macro')
        accuracy = balanced_accuracy_score(all_true, all_pred)
        precision = precision_score(all_true, all_pred, average='macro')
        recall = recall_score(all_true, all_pred, average='macro')

        print(f'Training results for epoch {epoch}')
        print('{:>8} | {:>8} | {:>9} | {:>6} | {:>5}'.format("f1-score", "accuracy", "precision", "recall", "loss"))
        print('{:>8.4f} | {:>8.4f} | {:>9.4f} | {:>6.4f} | {:>5.4f}'.format(f1, accuracy, precision, recall, tot_loss.detach().cpu().numpy()))

    return tot_loss
