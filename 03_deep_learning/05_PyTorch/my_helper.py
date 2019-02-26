#The project submission must include at least two files train.py and predict.py.
#The first file, train.py, will train a new network on a dataset and save the model as a checkpoint.
#The second file, predict.py, uses a trained network to predict the class for an input image.
#Feel free to create as many other files as you need.
#Our suggestion is to create a file just for functions and classes relating to the model => my_model.py
#and another one for utility functions like loading data and preprocessing images => my_helper.py
import torch
from torch import nn
from torchvision import models
import PIL
from PIL import Image
import numpy as np
import my_model
from my_model import Classifier_Network

def load_checkpoint(filepath):
    ''' Load a previously saved model

        Argument:
        ---------
        filepath: full path of checkpoint file

        return model, epochs, learning_rate, optimizer
    '''
    checkpoint = torch.load(filepath, map_location='cpu')
    model = getattr(models, checkpoint['deep_nn_type'])(pretrained=True)

    # Freeze parameters of the deep neural network so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = Classifier_Network(checkpoint['input_size'],
                                    checkpoint['output_size'],
                                    checkpoint['hidden_layers'],
                                    checkpoint['drop'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    optimizer = checkpoint['optimizer']
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.cpu()

    return model, epochs, learning_rate, optimizer


def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    width, height = img.size # get image size

    shortest_side = 256 # define shortest side as per requirement
    ratio = max(shortest_side/width, shortest_side/height) # calculate the ration
    new_size = int(width*ratio), int(height*ratio) # calculate the new_size taking into account the max of both ratio
    img.thumbnail(new_size, 3) # resize using thumbnail not sure this is the best way :-( There should be something better

    # caculate the box coordinate for the crop - why PIL is not having a centercrop like torchvision ???
    input_size = 224, 224 # size that the neural network is expecting
    left = (new_size[0] - input_size[0])/2
    top = (new_size[1] - input_size[1])/2
    right = (new_size[0] + input_size[0])/2
    bottom = (new_size[1] + input_size[1])/2
    crop_image = img.crop((left, top, right, bottom))

    np_image = np.array(crop_image)/255
    means = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    np_image = (np_image - means)/std_dev

    # pytorch expect the color channel to be the first one, not the third one like in PIL
    np_image = np_image.transpose((2,0,1))

    return np_image


def predict(image_path, model, cat_to_name, k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    # to perform the predict, we need a forward pass through the full network, deep part + classifier
    # first, we'll prepare the picture with the created function above
    pil_image = Image.open(image_path)
    np_image = process_image(pil_image)
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor) # if not type.floattensor, error because of .DoubleTensor

    # oh my god  : https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    # I've lost so much time because of this unsqueeze_ (inplace) add new dimension 1 at position 0
    tensor_image.unsqueeze_(0)
    #print(tensor_image.shape)

    model.eval()
    with torch.no_grad():
        output = model.forward(tensor_image)

    # forward pass return a logsoftmax, so we need to get the exponential of it now
    ps = torch.exp(output)
    top_k = ps.topk(k)

    probs, classes = top_k[0].numpy().tolist()[0], top_k[1].numpy().tolist()[0]
    #print(probs)
    #print(classes)

    # Convert indices to classes
    # I'm french and in france, idx to class means, when you have an idx, it will give you a class
    # I kept the naming convetion proposed here for model.class_to_idx, but it is really missleading :-(
    # since my classes_to_idx, means; when you give a classe from the network output classes
    # it will give you an idx. That idx, will be used with cat_to_name defined at the beginning
    # and we just need to chain both...
    classes_to_idx = {val: key for key, val in model.class_to_idx.items()}
    top_k_idx = [classes_to_idx[classe] for classe in classes]
    flowers = [cat_to_name[idx] for idx in top_k_idx]

    return probs, classes, flowers