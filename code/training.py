import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()

        # Number of hidden 
        self.hidden_size = hidden_size

        #Number of Layers
        self.num_layers = num_layers

        # LSTM based on input_size, hidden_size and num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Taking the number of output values and producing an output based on num_classes
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        # Forward propagate LSTM (We just care about the output not the weights)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step giving you a value
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Load')
    parser.add_argument('-input', dest='input', type=str, required=True,  help='Training data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_file = args.input.split(".")[0]

    input_size = 19
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    batch_size = 100
    n_iters = 3000
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as torch tensor with gradient accumulation abilities
            images = images.view(-1, seq_dim, input_size).requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        # Iterate through test dataset
        for images, labels in test_loader:
            # Resize image
            images = images.view(-1, seq_dim, input_dim)

            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            correct += (predicted == labels).sum()

        accuracy = 100 * correct / float(total)

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(accuracy)) 

    torch.save(model.state_dict(), save_file+".pt")
