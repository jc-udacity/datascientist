# my personnal model file, based on fc_model from Udacity
# most of code is copy/paste from the Jupyter notebook created in part 1
import torch
from torch import nn
import torch.nn.functional as F


class Classifier_Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers

        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


# Implement a function for the validation pass - chapter inference and validation
def validation(model, validloader, criterion, device='cpu'):
    test_loss = 0
    accuracy = 0
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)

        #images.resize_(images.shape[0], 784)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


# functions coming from what I've learned in the transfert training chapter...
def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # tell the model this is time to learn/train
    model.train()

    # change to cuda
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.4f}.. ".format(running_loss/print_every),
                          "Valid Loss: {:.4}.. ".format(test_loss/len(validloader)),
                          "Valid Accuracy: {:.4f}".format(accuracy/len(validloader)))

                    running_loss = 0

                    #make sure training is back on
                    model.train()
