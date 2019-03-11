from my_helper import load_checkpoint, predict
import argparse
import os.path
import json


parser = argparse.ArgumentParser(description='predict category of image, based on pre-trained model saved in checkpoint')
#positional argument full path to image
parser.add_argument('image_path', help='full image path used for inference')
# positional argument checkpoint to be used to build the model
parser.add_argument('checkpoint_path', help='full path to checkpoint')
#optional argument top_k
parser.add_argument('--top_k', type=int, help='Top K most likely classes', default=3)
# category to name
parser.add_argument('--category_names', help='category name json file', default='cat_to_name.json')
# GPU mode
parser.add_argument('--gpu', help='enable GPU mode for inference (disabled by default)', action='store_true')

args = parser.parse_args()

if os.path.isfile(args.image_path):
    if os.path.isfile(args.checkpoint_path):
        if os.path.isfile(args.category_names):
            # build the model.
            #Note, that I've not changed the returned elements since they might be useful
            #in later version if I want to improve learning, sarting again from a checkpoint
            model, epochs, learning_rate, optimizer = load_checkpoint(args.checkpoint_path, args.gpu)
            #print(model)

            # perform inference
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
            probs, classes, flowers = predict(args.image_path, model, cat_to_name, args.top_k)

            print(len(flowers))
            # print results
            # retrieve max flower name length
            maxlen = len(max(flowers, key=len))
            space = max(0, maxlen - 13 + 1)
            print('-' * (30+space))
            print('| flower name {0:{1}} | probability |'.format('', space))
            print('-' * (30+space))
            for i in range(args.top_k):
                print('| {0:{1}} | {2:.3f}       |'.format(flowers[i], max(13, maxlen), probs[i]))
            print('-' * (30+space))
