#!/bin/bash
# -*- coding: utf-8 -*-
import os
import numpy as np 
import matplotlib.pyplot as plt
import mogp_emulator
# import dill
# Load internal functions
# from data_prep import *
# from dgp import *
# from prep_UM import *

def plot(mu,var,testy,figname):
    x = np.arange(0, len(testy))
    plt.figure(figsize=(15,7))
    # else:
    s=np.sqrt(var)
    plt.errorbar(x, mu, 2*s, fmt='ok', lw =0.5)
    plt.scatter(x,testy,color='red',lw=0.02,alpha=0.8, label = 'True Std')

    plt.xlabel('Testing sample', fontsize = 20)
    plt.ylabel('STD', fontsize= 20)
    plt.legend(fontsize = 20)
    plt.savefig(figname, format='png', bbox_inches='tight')
    plt.show()

    return

def splitting(X, Y):
    number_of_rows = X.shape[2]
    test = 10
    train = number_of_rows-test
    # test = int(percent*number_of_rows*0.25)
    random_indices = np.random.choice(number_of_rows, size=train, replace=False)
    # # np.save(os.path.join(output_folder,'training_sound.npy'), random_indices)
    # random_indices = np.load(os.path.join(output_folder,'training_sound.npy'))
    X_train = X[:,:,random_indices]
    Y_train = Y[:,:,random_indices]
    random_indices = np.random.choice(number_of_rows, size=test, replace=False)
    # np.save(os.path.join(output_folder,'testing_sound.npy'), random_indices)
    # random_indices = np.load(os.path.join(output_folder,'testing_sound.npy'))
    X_test = X[:,:,random_indices]
    Y_test = Y[:,:,random_indices]

    return X_train, X_test, Y_train, Y_test

def main():
    EPS = 1.0
    output_folder = "../mogp_results/"
    X = np.load('../test_files/input.npy')
    Y = np.load('../test_files/target.npy')

    X_train, X_test, y_train, y_test = splitting(X, Y)
    print("Reshaping arrays")
    print(y_train.shape)
    X_train = np.reshape(X_train, (210,len(y_train[0,0,:])))
    X_train = X_train.T
    X_test = np.reshape(X_test, (210,len(y_test[0,0,:])))
    X_test = X_test.T
    y_train = np.reshape(y_train, (210,len(y_train[0,0,:])))
    y_test = np.reshape(y_test, (210,len(y_test[0,0,:])))
    # y_train = y_train.T
    # dr_tuned, loss = mogp_emulator.gKDR.tune_parameters(X_train, y_train,
                                                    # mogp_emulator.fit_GP_MAP,
                                                    # cXs=[3.], cYs=[3.])
    # print("Number of inferred dimensions is {}".format(dr_tuned.K))
    # dr_tuned = mogp_emulator.gKDR(X_train, y_train, K=4)
    # # Get number of inferred dimensions (usually gives 2)
    # if mogp:
        # # use object to create GP
        # gp_tuned = mogp_emulator.fit_GP_MAP(dr_tuned(X_train), y_train, kernel='Matern52')
    print(X_train.shape, y_train.shape)
    indices = [0,70,140]
    gp_tuned = mogp_emulator.fit_GP_MAP(X_train, y_train[indices,:], kernel='SquaredExponential')
    means, variances, d = gp_tuned.predict(X_test)
    for m, a, v in zip(means, y_test[indices,:], variances):
        print("Predicted mean: {} Actual mean: {} Variance: {}".format(m, a, v))
           
    #figname = os.path.join(output_folder, "mogp_sexp.png")
    #plot(means, variances, y_test[indices,:], figname)
    
    return

if __name__ == '__main__':
    main()

