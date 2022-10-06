import torch
import torch.nn as nn
# Torchvision module contains various utilities, classes, models and datasets 
# used towards computer vision usecases
from torchvision import datasets
from torchvision import transforms
from mlp_model_relu01 import MNIST_MLP_RELU1
from mlp_model_relu02 import MNIST_MLP_RELU2
from mlp_model_relu03 import MNIST_MLP_RELU3
import matplotlib.pyplot as plt

def mnist_loader(batch_size=512, classes=None):
    transform=transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_valid = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Select the classes which you want to train the classifier on.
    if classes is not None:
        mnist_train_idx = (mnist_train.targets == -1)
        mnist_valid_idx = (mnist_valid.targets == -1)
        for class_num in classes:
            mnist_train_idx |= (mnist_train.targets == class_num)
            mnist_valid_idx |= (mnist_valid.targets == class_num) 
        
        mnist_train.targets = mnist_train.targets[mnist_train_idx]
        mnist_valid.targets = mnist_valid.targets[mnist_valid_idx]
        mnist_train.data = mnist_train.data[mnist_train_idx]
        mnist_valid.data = mnist_valid.data[mnist_valid_idx]
    
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
    mnist_valid_loader = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, shuffle=True, num_workers=1)
    return mnist_train_loader, mnist_valid_loader

def main():
    # Load Data and Creat Data Loader based on Batch Size
    batch_size = 512 # Reduce this if you get out-of-memory error
    mnist_train_loader, mnist_valid_loader = mnist_loader(batch_size=batch_size)

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # The model
    model1 = MNIST_MLP_RELU1(layer_sizes=[784, 10])
    model1.set_device(device)
    
    #model 2
    model2 = MNIST_MLP_RELU2(layer_sizes=[784, 10])
    model2.set_device(device)
    
    # model 3
    model3 = MNIST_MLP_RELU3(layer_sizes=[784, 10])
    model3.set_device(device)
    
    # Our loss function and Optimizer
    model1.criterion = nn.CrossEntropyLoss()
    model1.optimizer = torch.optim.Adam(model1.parameters(), lr=0.0001) #lr is the learning_rate

    model2.criterion = nn.CrossEntropyLoss()
    model2.optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001) #lr is the learning_rate

    model3.criterion = nn.CrossEntropyLoss()
    model3.optimizer = torch.optim.Adam(model3.parameters(), lr=0.0001) #lr is the learning_rate

    # Train model for 2 epochs
    tlh1, tah1, vlh1, vah1 = model1.fit(mnist_train_loader, num_epochs=20, mnist_valid_loader=mnist_valid_loader)
    tlh2, tah2, vlh2, vah2 = model2.fit(mnist_train_loader, num_epochs=20, mnist_valid_loader=mnist_valid_loader)
    tlh3, tah3, vlh3, vah3 = model3.fit(mnist_train_loader, num_epochs=20, mnist_valid_loader=mnist_valid_loader)

    # Plot the results as a graph and save the figure.
    plt.figure()
    #plt.plot(tah, label='Train Accuracy')
    plt.plot(vah1, label='(784,5,5,10)')
    plt.plot(vah2, label='(784,20,20,10)')
    plt.plot(vah3, label='(784,50,50,10)')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.tight_layout()
    #plt.savefig('Intial_Run.pdf')
    
    
if __name__=='__main__':
    main()