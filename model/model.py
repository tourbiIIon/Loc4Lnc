import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt;
from sklearn.metrics import confusion_matrix, classification_report,f1_score,accuracy_score
from sklearn.metrics import classification_report
import torch.nn.functional as F

test_seq = torch.load('../feature/feature_independent_all.pt',map_location=torch.device('cpu'))
test_seq = torch.stack(test_seq).squeeze()
test_label = torch.load('../feature/label_independent_all.pt',map_location=torch.device('cpu'))

test_x = test_seq
test_y = test_label

class MyDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
class TextCNN(nn.Module):
    def __init__(self, num_classes):
        super(TextCNN, self).__init__()
        self.num_classes = num_classes
        self.num_filters = 100  
        self.filter_sizes = [3, 4, 5] 
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, 3072)) for k in self.filter_sizes]
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a dimension to conform to Conv2d's input requirements [batch_size, channels, height, width]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # Apply convolution and activation
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # pooling
        x = torch.cat(x, 1)  # Concatenate in the second dimension to form a long vector
        x = self.dropout(x) 
        logit = self.fc(x)  
        return logit

# parameters in model
batch_size = 128
test_set = MyDataset(test_x,test_y)
test_loader = DataLoader(test_set, batch_size=batch_size)

model = TextCNN(num_classes=5) 

checkpoint_path = 'ckpt_best.pth' 
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['net'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

test_preds = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_labels = np.array(test_labels)
test_preds = np.array(test_preds)

interested_labels1 = [0, 2, 3, 4]
interested_labels2 = [0, 2, 4]
interested_labels3 = [1, 2, 4]

mask1 = np.isin(test_labels, interested_labels1)
mask2 = np.isin(test_labels, interested_labels2)
mask3 = np.isin(test_labels, interested_labels3)

filtered_labels1 = test_labels[mask1]
filtered_labels2 = test_labels[mask2]
filtered_labels3 = test_labels[mask3]
filtered_preds1 = test_preds[mask1]
filtered_preds2 = test_preds[mask2]
filtered_preds3 = test_preds[mask3]

selected_labels1 = [0, 2, 3, 4]
selected_labels2 = [0, 2, 4]
selected_labels3 = [1, 2, 4]

accuracy1 = np.sum(filtered_labels1 == filtered_preds1) / len(filtered_labels1)
print(f'Accuracy for labels 0, 2, 3, 4: {accuracy1:.3f}')
print(classification_report(filtered_labels1, filtered_preds1, labels=selected_labels1, digits=3))
accuracy2 = np.sum(filtered_labels2 == filtered_preds2) / len(filtered_labels2)
print(f'Accuracy for labels 0, 2, 4: {accuracy2:.3f}')
print(classification_report(filtered_labels2, filtered_preds2, labels=selected_labels2, digits=3))
accuracy3 = np.sum(filtered_labels3 == filtered_preds3) / len(filtered_labels3)
print(f'Accuracy for labels 1, 2, 4: {accuracy3:.3f}')
print(classification_report(filtered_labels3, filtered_preds3, labels=selected_labels3, digits=3))
