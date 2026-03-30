#train.py

#train and test the MLP model
import torch
import torch.nn as nn


from mlp_architecture import MLP    ###ich habe eingefuegt
from mlp import test, train, plot_metrics

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
# # save the weights of the model in mlp_mnist.pth
losses, accuracies = train(model, device, loss_F, optimizer, num_epochs, batch_size, modelName='mlp_mnist_1')
plot_metrics(losses, accuracies, num_epochs)   


# Load the trained model
model.load_state_dict(torch.load("mlp_mnist_1.pth"))
model.eval()

# # Test the model with MNIST test dataset
test(model, device)


