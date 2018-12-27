import argparse
from time import time
import json
import torch
from PIL import Image
import helper

valid_networks = {'vgg16': 25088, 'densenet121': 1024  }
# Define the argparser to get the arguments
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', dest='arch', default='vgg16', action='store')
    parser.add_argument('image', type=str, action='store', default='flowers/test/11/image_03115.jpg', help='directory and path to input image')
    parser.add_argument('checkpoint', type=str, action='store', default='checkpoint.pth', help='checkpoint for predcition model')
    parser.add_argument('--top_k', type=int, default=5, help='top K classes, default 5')
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help='training device ')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='categroy to names file')

    #parser.set_defaults(gpu=True)

    return parser.parse_args()

def main():
	input_args= get_input_args()
	model, class_to_idx = helper.load_checkpoint(input_args.checkpoint)
	model.class_to_idx = class_to_idx
	#print(model)
	topk = input_args.top_k
	gpu = input_args.gpu
	image = input_args.image

    #to do: welche input args brauche ich wirklich mindestens?
    # process input image, then use the model - output the image, that should go into the model
    #-> is implemented within the predict function

    # remarks about predict function :
    #  define gpu 
    # full path to image has to go in there
    #return classes probabilities from  prediction function
	[probs], classes = helper.predict(image, model, topk, gpu)
#   

	cat_to_name = helper.load_categories(input_args.category_names)
	# Output result (Hier werden die Classes und die wahrscheinlichkeiten rausgeschrieben)
	#print(cat_to_name, probs, classes)
	for a, b in zip(classes, probs):
		print("{} = {:.2f}%".format(cat_to_name[a], b*100))

if __name__ == "__main__":
	main()