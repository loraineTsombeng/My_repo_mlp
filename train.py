#train.py

#train and test the CNN model
import torch
import torch.nn as nn

from MyCnn import ConvNet
from myCnn3 import ConvNet as ConvNet3
from mlp import MLP    ###ich habe eingefuegt
from cnn import test, train, plot_metrics, run
from data import get_activation, load_emnist_mapping, save_activations, visualize_activations

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 20
#num_epochs = 30
batch_size = 256
learning_rate = 0.001

# Dictionary to store activations
activations = {}

#model = ConvNet().to(device)
#model = ConvNet3().to(device)
model = MLP().to(device)
loss_F = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# # Train the model
# modelName like MINIST-CNN, EMNIST-letters-CNN, EMNIST-balanced-CNN  ##in mlp_emnist werden die Gewichte der mlp gespeichert
losses, accuracies = train(model, device, loss_F, optimizer, num_epochs, batch_size, modelName='mlp_emnist') #hier und zeile darunter ver                                           #modelName='mlp_emnist' oder modelName=f'cnn_epoch{num_epochs}'
plot_metrics(losses, accuracies, num_epochs)   


# Load the trained model
model.load_state_dict(torch.load("mlp_emnist.pth"))  #hierauch EMNIST-balanced-CNN oder mlp_emnist.pth
model.eval()

# # Test the model with EMNIST test dataset
test(model, device)

# # Run the model

#register hooks to capture activations
# model.conv1.register_forward_hook(get_activation(activations, "conv1"))
# model.bn1.register_forward_hook(get_activation(activations, "bn1"))
# model.convStride1.register_forward_hook(get_activation(activations, "convStride1"))
# model.bn2.register_forward_hook(get_activation(activations, "bn2"))
# model.convStride2.register_forward_hook(get_activation(activations, "convStride2"))
# model.bn3.register_forward_hook(get_activation(activations, "bn3"))
# model.conv2.register_forward_hook(get_activation(activations, "conv2"))
# model.bn4.register_forward_hook(get_activation(activations, "bn4"))

# model.fc1.register_forward_hook(get_activation(activations, "fc1"))
# model.fc2.register_forward_hook(get_activation(activations, "fc2"))
# model.fc3.register_forward_hook(get_activation(activations, "fc3"))

# #forward pass a test image
# img_path = f"../TestData/L.png"
# pred, model = run(model, device, img_path)
# # output the predicted label
# # mapping of EMNIST labels
# mapping = load_emnist_mapping()
# print(f"Predicted label: {pred}")
# print(f"Predicted label: {mapping[pred]}")

# visualize_activations(activations["conv1"], "conv1")
# visualize_activations(activations["convStride1"], "convStride1")
# visualize_activations(activations["conv2"], "conv2")
# visualize_activations(activations["convStride2"], "convStride2")

# #visualize_activations(mapping[activations["flatten"]], "flatten")
# visualize_activations(activations["fc1"], "fc1")
# visualize_activations(activations["fc2"], "fc2")
# visualize_activations(activations["fc3"], "fc3")

# labeld so that they can be easily identified in the visualization
# save_activations(model.conv1_x, "1_conv1")
# save_activations(model.convStride1_x, "2_convStride1")
# save_activations(model.conv2_x, "3_conv2")
# save_activations(model.convStride2_x, "4_convStride2")

# save_activations(model.view_x, "5_flatten")
# save_activations(model.fc1_x, "6_fc1")
# save_activations(model.fc2_x, "7_fc2")
# save_activations(model.fc3_x, "8_fc3")

# for i in range(10):
#     if i == 4:
#         for j in range(1):
#             img_path = f"../TestData/{i}.png"
#             pred, model = run(model, device, img_path)
#             print(f"Image: {i}  Predicted label: {pred}")

#             visualize_activations(model.conv1_x, "conv1")
#             visualize_activations(model.convStride1_x, "convStride1")
#             visualize_activations(model.conv2_x, "conv2")
#             visualize_activations(model.convStride2_x, "convStride2")

#             visualize_activations(model.view_x, "flatten")
#             visualize_activations(model.fc1_x, "fc1")
#             visualize_activations(model.fc2_x, "fc2")
#             visualize_activations(model.fc3_x, "fc3")
#             #print(f"Testing image of digit {i}, run {j+1}")
#         j += 1
    
#     i += 1
