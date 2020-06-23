# Code by: Kayla Bollinger
# Date: 06/17/20
# Email: kbolling@andrew.cmu.edu

import os, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from model_aux import *

parser = argparse.ArgumentParser('MODEL')
parser.add_argument('--hidden_dim', type=int, default=30,
	help='number of units per hidden layer')
parser.add_argument('--nn_depth', type=int, default=1,
    help='number of hidden layers')
parser.add_argument('--batch_size', type=int, default=4,
	help='number of')
parser.add_argument('--num_epoch', type=int, default=5000,
	help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.01,
	help='learning rate')
parser.add_argument('--data_dir', type=str, default='data',
	help='name for data directory')
parser.add_argument('--log_dir', type=str, default='results',
	help='name for directory in which to save results')
args = parser.parse_args()

class MLP(nn.Module):
    '''
    Notation:
        h_0 = first layer of network
        h_last = last layer of network
        h_i (1 <= i <= last-1) can refer to the ith hidden layer
    '''

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()

        self.hidden.append(nn.Linear(input_dim, hidden_dim))
		
        for i in range(args.nn_depth - 1):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))

        self.hidden.append(nn.Linear(hidden_dim, output_dim))
        self.sigmoid = nn.ReLU()
        return

    def forward(self, x0):
        length = len(self.hidden)
        for i in range(length):
            layer = self.hidden[i]
            if i == length - 1:
            	x0 = layer(x0)
            else:
                x0 = self.sigmoid(layer(x0))
        return x0

if __name__ == "__main__":
	''' Make directory to save results. '''
	make_directory(args.log_dir)

	''' Save parameters for reference. '''
	with open(args.log_dir+'/args.txt', 'w') as f:
	    json.dump(args.__dict__, f, indent=2)

    #'''Load data.'''
	train_y = np.load(args.data_dir+'/train_y.npy')
	train_v = np.load(args.data_dir+'/train_v.npy')
	val_y = np.load(args.data_dir+'/val_y.npy')
	val_v = np.load(args.data_dir+'/val_v.npy')
	test_y = np.load(args.data_dir+'/test_y.npy')
	test_v = np.load(args.data_dir+'/test_v.npy')

	''' Define NET structure. '''
	dim = train_y.shape[2] # dimension of data
	net = MLP(input_dim=dim, hidden_dim=args.hidden_dim, output_dim=dim)
	optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 
	criterion = nn.MSELoss(reduction='mean')

	''' Train model. '''
	# toggle comment of next two lines to either train network, or run tests with code on already trained network
	train_nn(train_y,train_v,val_y,val_v,net,criterion,optimizer,args)
	# net.load_state_dict(torch.load(args.log_dir+'/net_state_dict.pt'), strict=False)
