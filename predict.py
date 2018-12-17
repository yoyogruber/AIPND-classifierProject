import argparse
from time import time
import json
import torch
from PIL import Image
import json
import helper

valid_networks = {'vgg16': 25088, 'densenet121': 1024  }
# Define the argparser to get the arguments
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', dest='arch', default='vgg16', action='store')
    parser.add_argument('image', type=str, default='flowers', help='directory for input image')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='checkpoint for predcition model')
    parser.add_argument('--top_k', type=int, default=5, help='top K classes, default 5')
    parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate, default 0.001')
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help='training device ')
    parser.add_argument('--cat_names', type=str, help='categroy to names file')

    parser.set_defaults(gpu=True)

    return parser.parse_args()

def main():
	input_args= get_input_args()
	model, class_to_idx, gpu = load_checkpoint(input_args.checkpoint)





#
#
#

	cat_to_name = load_categories(command_line_inputs.category_names)
	# Output result
	for i, j in zip(classes, probabilities):
	print("{} = {:.2f}%".format(cat_to_name[i], j*100))

if __name__ == "__main__":
    main()