import torch

from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate(model, data_loader, criterion, device, is_test=False, epoch=0, print_metrics=True):
    
    model.eval()
    
    tot_loss = 0
    pred_labels, true_labels = [], []
    
    with torch.no_grad():
        for i, (img, ts_features, label) in enumerate(tqdm(data_loader)):
            img, label = img.to(device), label.to(device)
       
            output = model(img, ts_features)
            loss = criterion(output, label)

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

        evaluation = 'Test' if is_test else 'Validation'

        print(f'{evaluation} results for epoch {epoch}')
        print('{:>8} | {:>8} | {:>9} | {:>6} | {:>5}'.format("f1-score", "accuracy", "precision", "recall", "loss"))
        print('{:>8.4f} | {:>8.4f} | {:>9.4f} | {:>6.4f} | {:>5.4f}'.format(f1, accuracy, precision, recall, tot_loss.detach().cpu().numpy()))

    if is_test and print_metrics:
        cf_matrix = confusion_matrix(all_true, all_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = ["Damage","No Damage"], columns = ["Damage","No Damage"])
        plt.figure(figsize = (12,7))
        sns.heatmap(df_cm, annot=True)
    
    return tot_loss