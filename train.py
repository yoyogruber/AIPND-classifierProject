import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image
import numpy as np
import json 
import helper
from workspace_utils import active_session

valid_networks = {'vgg16': 25088, 'densenet121': 1024  }
# Define the argparser to get the arguments
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', dest='arch', default='vgg16', action='store')
    parser.add_argument('data_dir', type=str, default='flowers', help='directory to load images')
    parser.add_argument('--save_dir', type=str, default='', help='directory, where to save checkpoints')
    parser.add_argument('--hidden_units', type=int, default=5024, help='hidden units, default 5024')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate, default 0.001')
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help='training device ')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs, default 10')

    #parser.set_defaults(gpu=True)

    return parser.parse_args()

def main():
	input_args= get_input_args()
	#print(input_args)
	if input_args.arch == 'vgg16' :
		input_size = valid_networks[input_args.arch]
		arch = input_args.arch
	elif input_args.arch == 'densenet121':
		input_size = valid_networks[input_args.arch]
		arch = input_args.arch    
		#return arch, input_size
	else:
		print('please enter a valid network vgg16 or densenet121')

	# use helper function load_data to create trainloader, validationloader and testlodaer
	trainloader, validationloader, testloader, class_to_idx, batch_size = helper.load_data(input_args.data_dir)
	#print(trainloader)
    


	# Start building the classifier:
	dropout = 0.5
	output_size = 102
	hidden_sizes = input_args.hidden_units

	if arch == 'vgg16':
		model = models.vgg16(pretrained = True)

		#return model, hidden_sizes
	elif arch == 'densenet121':
		model = models.densenet121(pretrained= True)

	# Freeze parameters 
	for param in model.parameters():
		param.requires_grad = False
	classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes)),
                      ('relu1', nn.ReLU()),
                      ('dropout1', nn.Dropout(dropout)),
                      ('output', nn.Linear(hidden_sizes, output_size)),
                      ('softmax', nn.LogSoftmax(dim=1))
	]))
	model.classifier = classifier
	#print(model)
	#criterion
	criterion = nn.NLLLoss()
	learnrate = input_args.learning_rate
	optimizer = optim.Adam(model.classifier.parameters(), learnrate)

	#device
	if input_args.gpu :
		device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
		#return device
	else:
		device = torch.device('cpu')
	#print(device)
	# training and printing the accuracy
	epochs = input_args.epochs
	steps = 0
	running_loss = 0
	print_every = 40
	print('training started')
	with active_session():
	    for e in range(epochs):
	        model.to(device)
	        for images, labels in trainloader:
	            steps += 1
	            images, labels = images.to(device), labels.to(device)


	            optimizer.zero_grad()

	            output = model.forward(images)
	            loss = criterion(output, labels)
	            loss.backward()
	            optimizer.step()

	            running_loss += loss.item()

	            if steps % print_every == 0:
	                # Make sure network is in eval mode for inference
	                model.eval()

	                # Turn off gradients for validation, saves memory and computations
	                with torch.no_grad():
	                    validation_loss, accuracy = helper.validation(model, validationloader, criterion, device)

	                print("Epoch: {}/{}.. ".format(e+1, epochs),
	                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
	                      "Valid Loss: {:.3f}.. ".format(validation_loss/len(validationloader)),
	                      "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))

	                running_loss = 0

	                # Make sure training is back on
	            model.train()
	print('training  model finished')
	# Do validation on test set
	with torch.no_grad():
		test_loss, accuracy = helper.validation(model, testloader, criterion, device)

	print("Test-Loss: {}\n".format(test_loss/len(testloader)),
    "Test-Accuracy: {}".format(accuracy/len(testloader)))

    # : Save the checkpoint
	save_location = input_args.save_dir  + 'checkpoint.pth'
	checkpoint = {
    'arch': input_args.arch,
    'input_size': model.classifier[0].in_features,
    'state_dict': model.classifier.state_dict(),
    'class_to_idx': class_to_idx,
    'output_size': output_size,
    'classifier': model.classifier,
    'hidden_layers': hidden_sizes,
    'dropout': dropout,
    'optimizer state': optimizer.state_dict,
    'number of epochs': epochs,
    'gpu': input_args.gpu
	}

	torch.save(checkpoint, save_location)
	print('saved checkpoint')

if __name__ == "__main__":
    main()