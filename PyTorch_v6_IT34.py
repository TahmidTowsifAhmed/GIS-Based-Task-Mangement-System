import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve
import pickle
import matplotlib.pyplot as plt

# PyTorch Dataset defined
class TaskDataset(Dataset):
    def __init__(self, data, vectorizer, label_encoder, stop_words):
        self.data = data
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.stop_words = stop_words
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['title']+' '+ self.data.iloc[idx]['description']
        text = ' '.join([word for word in text.split() if word.lower() not in self.stop_words])
        label = self.label_encoder.transform([self.data.iloc[idx]['first_assigned']])[0]
        features = self.vectorizer.transform([text]).toarray()[0]
        return torch.tensor(features).float(), torch.tensor(label).long()
    
# Dataset loading
data = pd.read_csv('IT34_Internal_tasks_for_training.csv', encoding='ISO-8859-1')
data.fillna('',inplace=True)

# Splitting into train and test dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Load stop words from file
with open('stopwords.txt', 'r', encoding='ISO-8859-1') as f:
    stop_words = [line.strip() for line in f]

# LabelEncoder fitting on the combined data
combined_data = pd.concat([train_data, test_data], axis=0)
le = LabelEncoder()
le.fit(combined_data['first_assigned'])

# Vectorizing the dataset
vectorizer = CountVectorizer(stop_words=stop_words)
vectorizer.fit(train_data['title'] + ' ' + train_data['description'])

# PyTorch datasets and dataloaders
train_dataset = TaskDataset(train_data, vectorizer, le, stop_words)
test_dataset = TaskDataset(test_data, vectorizer, le, stop_words)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# The Neural Network
class TaskClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TaskClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Model Initializing
model = TaskClassifier(input_size=len(vectorizer.vocabulary_), hidden_size=100, output_size=len(le.classes_))

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Model Training
losses = []
accuracies = []
for epoch in range(50):
    running_loss =0.0
    total_correct = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
    train_loss = running_loss/len(train_dataset)
    train_accuracy = total_correct / len(train_dataset)
    losses.append(train_loss)
    accuracies.append(train_accuracy)
    print('Epoch: {} | Training Loss: {:.6f} | Training Accuracy: {:.2%}'.format(epoch + 1, train_loss, train_accuracy))

"""
# Plot the epoch and accuracy vs Training loss graph 
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), losses, label='Training Loss')
plt.plot(range(1, 51), accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy Curve')
plt.legend()
plt.savefig('loss_accuracy_curve.png')
plt.show()
"""

# Plot the epoch vs Training loss graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curve')
plt.savefig('loss_curve_IT34.png')
plt.show()

# Plot the epoch vs Training accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), accuracies)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy Curve')
plt.savefig('accuracy_curve_IT34.png')
plt.show()

# Obtain predicted probabilities for each class
pred_probs = []
true_labels = []

model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        pred_probs.extend(probabilities.tolist())  # Store predicted probabilities for each sample
        true_labels.extend(labels.tolist())  # Store true labels for each sample

# Convert the lists to numpy arrays
pred_probs = np.array(pred_probs)
true_labels = np.array(true_labels)

# Compute precision and recall for each class
precision = dict()
recall = dict()

for class_idx, class_name in enumerate(le.classes_):
    class_pred_probs = pred_probs[:, class_idx]  # Predicted probabilities for the current class
    class_true_labels = (true_labels == class_idx).astype(int)  # True labels for the current class (binary)

    precision[class_name], recall[class_name], _ = precision_recall_curve(class_true_labels, class_pred_probs)

# Plot precision-recall curves for each class
plt.figure(figsize=(8, 6))
for class_name in le.classes_:
    plt.plot(recall[class_name], precision[class_name], label=f"Class {class_name}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Each Class")
plt.savefig("precision_recall_curve_IT34.png")
plt.show()

# Plot labels for each class
plt.figure(figsize=(6, 4))
for class_name in le.classes_:
    plt.plot(0, 0, label=f"Class {class_name}")

plt.legend(loc="center")
plt.axis("off")
plt.savefig("labels_IT34.png")
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'trained_model_IT34.pkl')

# Save the vectorizer
with open('vectorizer_IT34.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Load the saved model's state dictionary
state_dict = torch.load('trained_model_IT34.pkl')

# Create an instance of the TaskClassifier class
model = TaskClassifier(input_size=len(vectorizer.vocabulary_), hidden_size=100, output_size=len(le.classes_))

# Load the saved state dictionary
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Load the CSV file
new_tasks = pd.read_csv('IT34_Internal_tasks_for_predictions.csv', encoding='ISO-8859-1')
new_tasks.fillna('', inplace=True)
print(new_tasks)

# Concatenate the title and description columns
new_tasks['text'] = new_tasks['title'] + ' ' + new_tasks['description']

# Transform the text column into a feature vector
new_features = torch.tensor(vectorizer.transform(new_tasks['text']).toarray()).float()

# Make predictions using the trained model
with torch.no_grad():
    outputs = model(new_features)
    _, predicted = torch.max(outputs.data, 1)
    predicted_labels = le.inverse_transform(predicted.numpy())

# Add the predicted labels to the new_tasks DataFrame
new_tasks['predicted_label'] = predicted_labels

# Save the new_tasks DataFrame to a new CSV file
new_tasks.to_csv('IT34_Internal_tasks_with_predictions.csv', index=False, sep=';')

