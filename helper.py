import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json


def load_data(data_dir):
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'
	# Define your transforms for the training, validation, and testing sets

	train_transforms = transforms.Compose([transforms.Resize(224),
	                                      transforms.RandomRotation(30),
	                                      transforms.RandomResizedCrop(224),
	                                      transforms.RandomHorizontalFlip(), 
	                                      transforms.ToTensor(),
	                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
	validation_transforms = transforms.Compose([transforms.Resize(255),
	                                      transforms.CenterCrop(224),
	                                      transforms.ToTensor(),
	                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
	test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
	                                      transforms.CenterCrop(224),
	                                      transforms.ToTensor(),
	                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

	# Load the datasets with ImageFolder
	train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
	validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
	test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
	# Using the image datasets and the trainforms, define the dataloaders
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) 
	validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True) 
	testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

	class_to_idx = train_data.class_to_idx

	return trainloader, validationloader, testloader, class_to_idx, trainloader.batch_size

# function for the validation pass
def validation(model, validationloader, criterion, device):
    validation_loss = 0
    accuracy = 0
    
    for images, labels in validationloader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        validation_loss = criterion(output, labels).item()
       
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validation_loss, accuracy

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    #model=models.vgg16(pretrained=True)
    #print(model)
    classifier = checkpoint['classifier']
    model.classifier = classifier
    model.classifier.load_state_dict(checkpoint['state_dict'])
    gpu =checkpoint['gpu']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, checkpoint['class_to_idx'] #, gpu
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
  needs to be corrected along these lines:
  def process_image(image):
  width, height = image.size
  size = 256, 256
  if width > height:
      ratio = float(width) / float(height)
      newheight = ratio * size[0]
      image = image.resize((size[0], int(floor(newheight))), Image.ANTIALIAS)
  else:
      ## Calculate for the other case
    '''
    # Open the image and resize it
    image_pil = Image.open(image)
    image_pil = image_pil.resize((256,256))
    
    # Resize - Adapted from https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    width, height = image_pil.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    image_pil = image_pil.crop((left, top, right, bottom))
    
    # Turn into a np_array
    np_image = np.array(image_pil)/255
    
    # Undo mean and std and then transpose the array
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])  
    np_image = (np_image - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def load_categories(path='cat_to_name.json'):
	cat_to_name = None
	with open(path, 'r') as f:
		cat_to_name = json.load(f)

	return cat_to_name


def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    #check fo gpu
    if gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # If gpu used, check for cuda
    else:
            device = 'cpu'
    class_to_idx = model.class_to_idx
    #print(class_to_idx)
    # Inverting class_to_idx - Kudos to Mentor ShintoT
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    model= model.to(device)
    img_predict = torch.FloatTensor([process_image(image_path)])
    img_predict = img_predict.to(device)

    with torch.no_grad():
        result = model(img_predict)
    
    
    # Save probs and classes
    probs = np.array(torch.exp(result).topk(topk)[0])
    #probs = probs.to('cpu') # ensuring numpy conversion 
    classes = np.array(torch.exp(result).topk(topk)[1])
    classes = [idx_to_class[c] for c in classes[0,:]]  
    
    return(probs, classes)

