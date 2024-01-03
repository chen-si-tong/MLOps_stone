import argparse
import torch
import numpy as np
from torchvision import transforms
from model import MNISTModel
import pickle

def main(model_file, data_file):

    model =  MNISTModel()  
    model.load_state_dict(torch.load(model_file))
    model.eval()

   
    if data_file.endswith('.npy'):
        data = np.load(data_file)
    elif data_file.endswith('.pkl'):
       with open(data_file, 'rb') as file:
           data = pickle.load(file)
    else:

        pass


    with torch.no_grad():
        predictions = model(data)
    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="Pretrained model file path")
    parser.add_argument("data_file", help="Data file path (numpy or pickle)")

    args = parser.parse_args()
    main(args.model_file, args.data_file)
