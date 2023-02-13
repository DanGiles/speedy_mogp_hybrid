#!/bin/bash
# -*- coding: utf-8 -*-
import os
import numpy as np 
import matplotlib.pyplot as plt
import mogp_emulator
import dill
# Load internal functions
#from loop_files import *
# from data_prep import *
# from dgp import *
# from prep_UM import *

def plot(mu, testy, var, level, variable, figname):
    x = np.arange(0, len(testy))
    plt.figure(figsize=(15,7))
    # else:
    s=np.sqrt(var)
    plt.errorbar(x, mu, 2*s, fmt='ok', lw =0.5)
    plt.scatter(x,testy,color='red',lw=0.02,alpha=0.8, label = 'True Std')
    plt.title( 'Atmospheric Level %i'%level)
    plt.xlabel('Testing sample', fontsize = 20)
    plt.ylabel('$\sigma$ ('+ variable +')', fontsize= 20)
    plt.legend(fontsize = 20)
    plt.savefig(figname, format='png', bbox_inches='tight')
    plt.show(block = False)

    return

def splitting(X, Y):
    number_of_rows = X.shape[2]
    test = 40
    train = number_of_rows-test
    # test = int(percent*number_of_rows*0.25)
    random_indices = np.random.choice(number_of_rows, size=train, replace=False)
    # # np.save(os.path.join(output_folder,'training_sound.npy'), random_indices)
    # random_indices = np.load(os.path.join(output_folder,'training_sound.npy'))
    X_train = X[:,:,random_indices]
    Y_train = Y[:,:,random_indices]
    inverted_idx = [x not in random_indices for x in range(0, number_of_rows)]
    # random_indices = np.random.choice(number_of_rows, size=test, replace=False)

    # np.save(os.path.join(output_folder,'testing_sound.npy'), random_indices)
    # random_indices = np.load(os.path.join(output_folder,'testing_sound.npy'))
    X_test = X[:,:,inverted_idx]
    Y_test = Y[:,:,inverted_idx]

    return X_train, X_test, Y_train, Y_test

def splitting_spatial(X, Y, oro, ls):
    number_of_rows = X.shape[2]
    test = 50
    train = 300
    random_indices = np.random.choice(number_of_rows, size=train, replace=False)
    X_train = X[:,:,random_indices]
    Y_train = Y[:,:,random_indices]
    
    cell_indices = np.empty(len(random_indices), dtype=int)
    idx = 0
    for i in random_indices:
        cell_indices[idx] = i%196 
        idx +=1
    oro_train = oro[:, cell_indices]
    ls_train = ls[:, cell_indices]
    inverted_idx = [x not in random_indices for x in range(0, number_of_rows)]
    # random_indices = np.load(os.path.join(output_folder,'testing_sound.npy'))
    cell_indices = np.empty(test, dtype=int)
    idx = 0
    count = 0
    for i in inverted_idx:
        if i and idx < test:
            cell_indices[idx] = count%196
            idx += 1
        count += 1
    X_test = X[:,:,cell_indices]
    Y_test = Y[:,:,cell_indices]
    oro_test = oro[:, cell_indices]
    ls_test = ls[:, cell_indices]
    return X_train, X_test, Y_train, Y_test, oro_train, oro_test, ls_train, ls_test


def reorder(array, ld):
    new_array = np.zeros(np.shape(array))
    for j in range(len(array[0,:])):
        for i in range(ld):
            new_array[2*i,j] = array[i,j]
            new_array[2*i+1,j] = array[i+ld,j]
            # new_array[3*i+2,j] = array[i+(ld*2),j]

    return new_array

def reorder_speedy(array, ld):
    new_array = np.zeros(np.shape(array))
    for j in range(len(array[0,:])):
        for i in range(ld):
            new_array[3*i,j] = array[i,j]
            new_array[3*i+1,j] = array[i+ld,j]
            new_array[3*i+2,j] = array[i+(ld*2),j]

    return new_array



def crop_speedy(array):
    "Pressure Levels in Speedy defined on pressure levels"
    "30, 100, 200, 300, 500, 700, 850 and 925 hPa"

    pressure_level = [3000, 10000, 20000, 30000, 50000, 70000, 85000, 92500]
    # pressure_level *= 100
    # for ex in range(len(array[0,0,:])):
    indices = np.zeros(len(pressure_level), dtype=int)
    for j in range(len(pressure_level)):
        indices[j] = np.argmin(np.abs(pressure_level[j]-array[0,:,0]))
    return indices

def chunking(X, oro, ls, layer, y):
    train = np.empty((X.shape[0], 13))
    target = np.empty((3, X.shape[0]))

    if layer == 0:
        train[:,0:9] = X[:,0:9]
        train[:,9:11] = oro[:,:]
        train[:,11:] = ls[:,:]
        target[:,:] = y[:3,:]
    elif (3*(layer-1))+9 > len(X[0,:]):
        train[:,0:9] = X[:,-9:]
        train[:,9:11] = oro[:,:]
        train[:,11:] = ls[:,:]
        target[:,:] = y[-3:,:]
    else:
        train[:,0:9] = X[:,(3*(layer-1)):(3*(layer-1))+9]
        train[:,9:11] = oro[:,:]
        train[:,11:] = ls[:,:]
        target[:,:] = y[(3*(layer)):(3*(layer))+3,:]
    return train, target

def chunking_speedy(X, X_ps, oro, ls, layer, y):
    train = np.empty((9, X.shape[2]))
    target = np.empty((2, X.shape[2]))

    if layer == 0:
        train[0,:] = X_ps*np.ones(X.shape[2])
        train[1:4,:] = X[0,:3,:]
        train[4:7,:] = X[1,:3,:]
        train[7,:] = oro[0,:]
        train[8,:] = ls[0,:]
        target[:,:] = y[:,0,:]
        # target[1,:] = y[1,:3,:]
    elif layer >= 7:
        train[0,:] = X_ps*np.ones(X.shape[2])
        train[1:4,:] = X[0,-3:,:]
        train[4:7,:] = X[1,-3:,:]
        train[7,:] = oro[0,:]
        train[8,:] = ls[0,:]
        target[:,:] = y[:,-1,:]
    else:
        train[0,:] = X_ps*np.ones(X.shape[2])
        train[1:4,:] = X[0,(layer-1):(layer+2),:]
        train[4:7,:] = X[1,(layer-1):(layer+2),:]
        train[7,:] = oro[0,:]
        train[8,:] = ls[0,:]
        target[:,:] = y[:,layer,:]
    return train, target

def main():
    output_folder = "../mogp_speedy_arrays/"
    spatial = True
    speedy = True
    path = os.getcwd()
    print(path) 
    X = np.load('processed/mean.npy')
    Y = np.load('processed/std.npy')
    print("Loaded in the X and Y")
    if spatial:
        oro = np.load('processed/orography.npy')
        land_sea = np.load('processed/land_sea.npy')
        X_train, X_test, y_train, y_test, oro_train, oro_test, ls_train, ls_test = splitting_spatial(X, Y, oro, land_sea)
    else:
        X_train, X_test, y_train, y_test = splitting(X, Y)
    print(X_train.shape)

    if speedy:
        print("Cropping arrays")
        indices = crop_speedy(X_train)
        X_train_ps = X_train[0,0,:]
        X_test_ps = X_test[0,0,:]

        X_train = X_train[1:,indices,:]
        X_test = X_test[1:,indices,:]
        y_train = y_train[1:,indices,:]
        y_test = y_test[1:,indices,:]
    else:
        print("Reshaping arrays")
        indices = np.arange(0,70,1)
        X_train = np.reshape(X_train, (210,len(y_train[0,0,:])))
        X_test = np.reshape(X_test, (210,len(y_test[0,0,:])))
        y_train = np.reshape(y_train, (210,len(y_train[0,0,:])))
        y_test = np.reshape(y_test, (210,len(y_test[0,0,:])))
        print("Reordering arrays")
        ld = 70
        X_train = reorder(X_train, ld)
        X_test = reorder(X_test, ld)
        y_train = reorder(y_train, ld)
        y_test = reorder(y_test, ld)
        print("Transpose X")
        X_test = X_test.T
        X_train = X_train.T

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, oro_test.shape)  

    input_layers = 9
    variables = ['T', 'Q']
    for level in range(len(indices)):
        if spatial:
            # Setting up the training dataset
            input, target = chunking_speedy(X_train, X_train_ps, oro_train, ls_train, level, y_train)
            print("input and target", input.shape, target.shape)
            # Defining and fitting the MOGP
            gp = mogp_emulator.MultiOutputGP(input.T, target, kernel="SquaredExponential")
            gp = mogp_emulator.fit_GP_MAP(gp)
            # Save the trained mogp
            dill.dump(gp, open(os.path.join(output_folder, "gp%i.pkl"%level),"wb"))
            # Setting up the testing dataset
            test, truth = chunking_speedy(X_test, X_test_ps, oro_test, ls_test, level, y_test)
            print("test and truth", test.shape, truth.shape)
            # Loading the trained mogp from file. Not needed but used to test implementation
            gp = dill.load(open(os.path.join(output_folder, "gp%i.pkl"%level), "rb"))
            # Predict using the MOGP
            means, variances, d = gp.predict(test.T)

        else:
            # gp = mogp_emulator.fit_GP_MAP(X_train[:,(level*input_layers):((level+1)*input_layers)], y_train[level*3:(level+1)*3,:], kernel='SquaredExponential')
            means, variances, d = gp.predict(X_test[:,(level*input_layers):((level+1)*input_layers)])

        for m, a, v, s in zip(means, truth, variances, variables):
            print("Predicted mean: {} Actual mean: {} Variance: {}".format(m, a, v))
            figname = os.path.join(output_folder, "mogp_"+s+"_%i.png"%level)
            plot(m, a, v, level, s, figname)

    #plt.show()
    return

if __name__ == '__main__':
    main()

