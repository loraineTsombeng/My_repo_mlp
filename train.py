#train.py

#train and test the CNN model
import torch
import torch.nn as nn


from mlp import MLP    ###ich habe eingefuegt
from cnn import test, train, plot_metrics
from data import get_activation, load_emnist_mapping, save_activations, visualize_activations

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 30
batch_size = 256
learning_rate = 0.001

# Dictionary to store activations
activations = {}


model = MLP().to(device)
loss_F = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# # Train the model
# # save the weights of the model in mlp_emnist.pth
losses, accuracies = train(model, device, loss_F, optimizer, num_epochs, batch_size, modelName='mlp_emnist_1') 
plot_metrics(losses, accuracies, num_epochs)   


# Load the trained model
model.load_state_dict(torch.load("mlp_emnist_1.pth"))  
model.eval()

# # Test the model with EMNIST test dataset
test(model, device)


