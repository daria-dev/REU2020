import numpy as np
import argparse, json

'''
    Given directory of raw data splits into taining, validation and testing.
    To make this script general for arbitrary data sets, V_ss and T_ss arguments 
    have to be provided and have no default value.

    T and T_ss arguments are interchangeable but T is prioritized if provided.
'''

parser = argparse.ArgumentParser('DATA SETS')
parser.add_argument('--T', type=int, default=0,
	help='time point cutoff for training data')
parser.add_argument('--T_ss', type=int,
	help='number of points in training set')
parser.add_argument('--V_ss', type=int, 
    help='number of trajectories in validation set')
parser.add_argument('--data_dir', type=str, default='data',
		help='name for raw data directory')
parser.add_argument('--velocity_data', type=int, default = 0,
        help='true if raw data directory contains velocity data (0 - false, 1 - true)')
args = parser.parse_args()


if __name__ == "__main__":

    # create array of random indecies to shuffle both position and velocity
    # data sets equally
    raw_data_y = np.load(args.data_dir +'/data_y.npy')
    idx = np.random.permutation(raw_data_y.shape[0])

    raw_data_y = raw_data_y[idx]

    # make sets for velocity data if provided
    if (args.velocity_data == 1):
        raw_data_v = np.load(args.data_dir +'/data_v.npy')
        raw_data_v = raw_data_v[idx]

        # if time cut off given prioritize over number of training points
        if (args.T != 0):
            train_v = raw_data_v[:,:args.T,:]
            val_v = raw_data_v[:args.V_ss,args.T:,:]
            test_v = raw_data_v[args.V_ss:, args.T:, :]
        
        else:
            train_v = raw_data_v[:,:args.T_ss,:]
            val_v = raw_data_v[:args.V_ss,args.T_ss:,:]
            test_v = raw_data_v[args.V_ss:, args.T_ss:, :]

        # save data
        np.save(args.data_dir+'/'+'train'+'_v', train_v)
        np.save(args.data_dir+'/'+'val'+'_v', val_v)
        np.save(args.data_dir+'/'+'test'+'_v', test_v)

    # if time cut off given prioritize over number of training points
    if (args.T != 0):
        train_y = raw_data_y[:,:args.T,:]
        val_y = raw_data_y[:args.V_ss,args.T:,:]
        test_y = raw_data_y[args.V_ss:, args.T:, :]

    else:
        train_y = raw_data_y[:,:args.T_ss,:]
        val_y = raw_data_y[:args.V_ss,args.T_ss:,:]
        test_y = raw_data_y[args.V_ss:, args.T_ss:, :]

    # save data
    np.save(args.data_dir+'/'+'train'+'_y', train_y)
    np.save(args.data_dir+'/'+'val'+'_y', val_y)
    np.save(args.data_dir+'/'+'test'+'_y', test_y)
    