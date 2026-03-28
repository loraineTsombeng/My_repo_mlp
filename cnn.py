import torch
import matplotlib.pyplot as plt

from data import batch_generator_augmented, load_mnist_cnn, load_emnist_mapping, prep_image, show_image

# #test loading data
# # get mnist training data
# images_cnn, labels_cnn = load_emnist_cnn()
# # print(images_cnn.shape)  # (112800, 1, 28, 28)
# # # print(labels_cnn.shape)  # (112800,)
# mapping = load_emnist_mapping()

# for i in range(20):
#     print(f"Label: {mapping[labels_cnn[i]]}")
#     show_image(images_cnn[i])  # show the first image in the dataset



def train(model, device, loss_fn, optimizer, num_epochs, batch_size, modelName="cnn"):
# Train the model
    
    # get mnist training data
    images_cnn, labels_cnn = load_mnist_cnn()
    # print(images_cnn.shape)  # (112800, 1, 28, 28)
    # print(labels_cnn.shape)  # (112800,)

    model.train()

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (batch_imgs, batch_lbls) in enumerate(batch_generator_augmented(images_cnn, labels_cnn, batch_size=batch_size, augment=True)):
            
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

def run(model, device, image_path):
# predict label for a single image

    # Test with a single image
    image = prep_image(image_path)
    if image is None:
        print("Failed to preprocess image.")
        return None
    
    #show_image(image[0])  # show the preprocessed image

    model.eval()
    with torch.no_grad():
        image = torch.from_numpy(image).float().to(device)  # (1, 1, 28, 28)
        # run: flatten images
        images = images.view(images.size(0), -1)
        output = model(image)
        pred = torch.argmax(output, dim=1)
        return pred.item(), model


def test(model, device):
# Test the model

    # get mnist test data
    images_cnn, labels_cnn = load_mnist_cnn(train=False)
    #print(images_cnn.shape)  # (18800, 1, 28, 28)
    #print(labels_cnn.shape)  # (18800,)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        images = torch.from_numpy(images_cnn).float().to(device)
        # test: flatten images
        images = images.view(images.size(0), -1)
        labels = torch.from_numpy(labels_cnn).to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        print(total)
        correct += (predicted == labels).sum().item()

        # # Uncomment to see misclassified examples
        #mapping = load_emnist_mapping()
        # for i in range(total):
        #     if predicted[i] != labels[i]:
        #         print(f'Label: {mapping[int(labels[i].item())]}\t| Prediction: {mapping[int(predicted[i].item())]}')

    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy   