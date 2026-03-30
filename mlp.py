import torch
import matplotlib.pyplot as plt

from data import batch_generator_augmented, load_mnist_mlp


def train(model, device, loss_fn, optimizer, num_epochs, batch_size, modelName="mlp"):
# Train the model
    
    # get mnist training data
    images_mlp, labels_mlp = load_mnist_mlp(train=True)
    # print(images_mlp.shape)  # (112800, 1, 28, 28)
    # print(labels_mlp.shape)  # (112800,)

    model.train()

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (batch_imgs, batch_lbls) in enumerate(batch_generator_augmented(images_mlp, labels_mlp, batch_size=batch_size, augment=True)):
            
            images = batch_imgs.to(device)    # (batch_size, 1, 28, 28)
            labels = batch_lbls.to(device)    # (batch_size,)
            # train: flatten images
            images = images.view(images.size(0), -1)
                    
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / (i + 1)
        accuracy = 100 * correct / total if total > 0 else 0

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%')

    print('Finished Training')
    torch.save(model.state_dict(), f'{modelName}.pth')
    return epoch_losses, epoch_accuracies

def plot_metrics(losses, accuracies, num_epochs):
# plot loss and accuracy curves
    epochs = range(1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    plt.figure()
    plt.plot(epochs, accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.show()



def test(model, device):
# Test the model

    # get mnist test data
    images_mlp, labels_mlp = load_mnist_mlp(train=False)
    #print(images_mlp.shape)  # (18800, 1, 28, 28)
    #print(labels_mlp.shape)  # (18800,)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        images = torch.from_numpy(images_mlp).float().to(device)
        # test: flatten images
        images = images.view(images.size(0), -1)
        labels = torch.from_numpy(labels_mlp).to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        print(total)
        correct += (predicted == labels).sum().item()

        # # Uncomment to see misclassified examples
        #mapping = load_mnist_mapping()
        # for i in range(total):
        #     if predicted[i] != labels[i]:
        #         print(f'Label: {mapping[int(labels[i].item())]}\t| Prediction: {mapping[int(predicted[i].item())]}')

    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy   