import argparse, json
import torch.nn as nn
import numpy as np

from model_aux import *
from test_aux import test_loss

parser = argparse.ArgumentParser('MODEL')
parser.add_argument('--hidden_dim', type=int, default=32,
    help='number of units per hidden layer')
parser.add_argument('--nn_depth', type=int, default=1,
    help='number of hidden layers')
parser.add_argument('--batch_size', type=int, default=32,
    help='number of')
parser.add_argument('--num_epoch', type=int, default=1000,
    help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.01,
    help='learning rate')
parser.add_argument('--Reg', type=str, default='none',
    help='reguarization argument')
parser.add_argument('--Lambda', type=float, default=0.01,
    help='reguarization weight')
parser.add_argument('--data_dir', type=str, default='data',
    help='name for data directory')
parser.add_argument('--log_dir', type=str, default='results',
    help='name for directory in which to save results')
parser.add_argument('--dt', type=float, default=0.00998,
    help='time step for RK4')
parser.add_argument('--jacobian_lambda', type=float, default=0.01,
    help='jacobian reguarization weight')
args = parser.parse_args()

class MLP(nn.Module):
    '''
    Notation:
        h_0 = first layer of network
        h_last = last layer of network
        h_i (1 <= i <= last-1) can refer to the ith hidden layer
    '''

    def __init__(self, input_dim, hidden_dim, output_dim, nn_depth):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()

        self.hidden.append(nn.Linear(input_dim, hidden_dim))

        for i in range(nn_depth - 1):
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

class RK4(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nn_depth, dt):
        super(RK4, self).__init__()
        self.f = MLP(input_dim, hidden_dim, output_dim, nn_depth)
        self.dt = dt
        return

    def forward(self, x0):
        x1 = self.dt * self.f(x0)
        x2 = self.dt * self.f(x0 + x1/2.)
        x3 = self.dt * self.f(x0 + x2/2.)
        x4 = self.dt * self.f(x0 + x3)
        out = x0 + x1/6. + x2/3. + x3/3. + x4/6.

        return out

if __name__ == "__main__":
    ''' Make directory to save results. '''
    make_directory(args.log_dir)

    ''' Save parameters for reference. '''
    with open(args.log_dir+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print('\nNote that dt should be calculated using (Tend-T0)/num_point. \nChange value manually if necessary. \n')

    #'''Load data.'''
    train_y = np.load(args.data_dir+'/train_y.npy')
    val_y = np.load(args.data_dir+'/val_y.npy')
    test_y = np.load(args.data_dir+'/test_y.npy')

    ''' Define NET structure. '''
    dim = train_y.shape[2] # dimension of data
    net = RK4(input_dim=dim, hidden_dim=args.hidden_dim, output_dim=dim, nn_depth=args.nn_depth, dt=args.dt)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')

    ''' Train model. '''
    # toggle comment of next two lines to either train network, or run tests with code on already trained network
    train_nn(train_y,val_y,net,criterion,optimizer,args)
    net.load_state_dict(torch.load(args.log_dir+'/net_state_dict.pt'), strict=False)

    # toggle comments for next two functions to either calculate loss with MSE or Relative Loss

    #MSE
    #def crit(y, y_):
        #return np.mean((y - y_) ** 2)
    
    #Relative Loss
    def crit(y, y_):
        return np.mean(abs((y_ - y)/y))

    print('test loss:', test_loss(net, crit, test_y))
