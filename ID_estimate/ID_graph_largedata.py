import sys,argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from sklearn.utils.graph import graph_shortest_path
import scipy.io
import os
import multiprocessing
import pickle
import math
import h5py
import numpy.linalg as LA
import matplotlib.font_manager as font_manager

import knn
import config

import pdb

def func(x,a,b,c):
    return a*np.log(np.sin(x/1*np.pi/2.))
         
def func2(x,a):
    return -a/2.*(x-1)**2

def func3(x,a,b,c):
    return np.exp(c)*np.sin(x/b*np.pi/2.)**a

def main():
    args = config.parse_args()

    print('Load data')
    data = load_data(args)
    nrof_sample = data.shape[0]
    dim = data.shape[1]
    del data

    if os.path.isdir(args.resfolder) is False:
        os.makedirs(args.resfolder)

    if args.if_dist_table:
        print('compute distance table......')
        get_distTable(args)
    if args.if_knn_matrix:
        print('convert distance table to knn matrix......')
        knn2matrix(args)
    if args.if_shortest_path:
        print('compute shortest path......')
        get_shortestpath(args, nrof_sample)
    if args.if_histogram:
        print('compute histogram......')
        histogram,param_dict = get_histogram(args, nrof_sample)
    else:
        with open(os.path.join(args.resfolder, 'histogram.pkl'), 'rb') as f:
            histogram = pickle.load(f)
        with open(os.path.join(args.resfolder, 'params.pkl'), 'rb') as f:
            param_dict = pickle.load(f)

    print('compute intrinsic dimensionality......')
    ID_graph(args, histogram, param_dict, (nrof_sample,dim))

def load_data(args):
    if args.data_filename.endswith('.mat'):
        data = scipy.io.loadmat(args.data_filename)
        data = data['feat']
    elif args.data_filename.endswith('.npy'):
        data = np.load(args.data_filename)
    elif args.data_filename.endswith('.npz'):
        data = np.load(args.data_filename)
        data = data['feat']
    else:
        print('Unknown Input Format!')
        sys.exit(0)
    return data

def get_distTable(args):
    obj = knn.KNN(128, args.dist_table_filename, args.data_filename, 
        args.dist_type, args.if_norm)
    obj.get_dist_multip()

def knn2matrix(args):
    filename = os.path.join(args.resfolder, 'kneighbors.npm')
    dist_table = np.load(args.dist_table_filename)
    nrof_sample = dist_table.shape[0]
    matrix = np.memmap(filename, dtype='float32', mode='w+', shape=(nrof_sample, nrof_sample))
    for i in range(nrof_sample):
        idx = dist_table[i,0,0:args.n_neighbors]
        idx = idx.astype(int)
        matrix[i,idx] = dist_table[i,1,0:args.n_neighbors]
    del dist_table
    del matrix
    print('Matrix and dist_table is deleted!')

def check_connect(shortest_path):
    num = 0
    for i in range(shortest_path.shape[0]):
        nonzero = np.count_nonzero(shortest_path[i,:])
        if nonzero >= shortest_path.shape[0]/2:
            num += 1
    if num < shortest_path.shape[0]/2:
        print('The neighbors graph is highly disconnected, increase K or Radius parameters')
        print(num)
        exit(2)
    return num

def get_shortestpath(args, nrof_sample):
    matrix_filename = os.path.join(args.resfolder, 'kneighbors.npm')
    knn_matrix = np.memmap(matrix_filename, dtype='float32', mode='r', 
        shape=(nrof_sample, nrof_sample))
    radius = knn_matrix.max()
    shortest_path = np.memmap(os.path.join(args.resfolder, 'shortest_path.npm'),
        dtype='float32', mode='w+', shape=(nrof_sample, nrof_sample))
    shortest_path[:] = graph_shortest_path(knn_matrix, directed=False).copy()
    del knn_matrix
    del shortest_path

    param_dict = {}
    param_dict['radius'] = radius
    with open(os.path.join(args.resfolder, 'params.pkl'), 'wb') as f:
        pickle.dump(param_dict, f)

def get_histogram(args, nrof_sample):
    shortest_path = np.memmap(os.path.join(args.resfolder, 'shortest_path.npm'),
        dtype='float32', mode='r', shape=(nrof_sample, nrof_sample))
    with open(os.path.join(args.resfolder, 'params.pkl'), 'rb') as f:
        param_dict = pickle.load(f)
    num_connect = check_connect(shortest_path)
    param_dict['num_connect'] = num_connect
    avg=np.mean(shortest_path[np.nonzero(np.triu(shortest_path,1))])
    std=np.std(shortest_path[np.nonzero(np.triu(shortest_path,1))])
    param_dict['avg'] = avg
    param_dict['std'] = std
    histogram = np.histogram(shortest_path[np.nonzero(np.triu(shortest_path,1))].astype('float64'), args.n_bins)
    del shortest_path
    with open(os.path.join(args.resfolder, 'histogram.pkl'), 'wb') as f:
        pickle.dump(histogram, f)
    with open(os.path.join(args.resfolder, 'params.pkl'), 'wb') as f:
        pickle.dump(param_dict, f)
    return histogram, param_dict

def ID_graph(args, h, param_dict, data_shape):
    n_neighbors = args.n_neighbors
    radius=args.radius
    MSA=False
    n_bins = args.n_bins
    rmax=args.r_max
    mm=-10000

    if radius>0. :  
        filename = os.path.join(args.resfolder, 'R'+str(radius))
    else  :
        filename = os.path.join(args.resfolder, 'K'+str(n_neighbors))

    radius = param_dict['radius']
    avg = param_dict['avg']
    std = param_dict['std']
    num_connect = param_dict['num_connect']

    #1
    dx=h[1][1]-h[1][0]


    font_size = 17
    font_prop = font_manager.FontProperties(size=font_size)
    axis_label_prop = font_manager.FontProperties(size=17)

    plt.figure(1)
    plt.plot(h[1][0:n_bins]+dx/2,h[0],'o-',label='Histogram')
    plt.xlabel('Geodesic Distance (r)', fontproperties=axis_label_prop)
    plt.ylabel('P(r)', fontproperties=axis_label_prop)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.legend(prop=font_prop)
    plt.grid(True)
    plt.savefig(filename+'_hist.pdf', bbox_inches = "tight")
    distr_x = []
    distr_y = []

    if rmax> 0 : 
        avg=rmax
        std=min(std,rmax/2)
        print('\nNOTE: You fixed r_max for the initial fitting, average will have the same value')
    else : 
        mm=np.argmax(h[0])
        rmax=h[1][mm]+dx/2

    if args.r_min>= 0 : print('\nNOTE: You fixed r_min for the initial fitting: r_min = ',args.r_min)
    if args.r_min== -1 : print('\nNOTE: You forced r_min to the standard procedure in the initial fitting')

    print('\nDistances Statistics:')
    print('Average, standard dev., n_bin, bin_size, r_max, r_NN_max:', avg , std, n_bins, dx, rmax, radius,'\np')

    tmp=1000000
    if(args.r_min>=0) : tmp=args.r_min
    elif(args.r_min==-1) : tmp=rmax-std

    if(np.fabs(rmax-avg)>std) :
        print('ERROR: There is a problem with the r_max detection:')
        print('       usually either the histogram is not smooth enough (you may consider changing the n_bins with option -b)')
        print('       or r_max and r_avg are too distant and you may consider to fix the first detection of r_max with option -M' )
        print('       or to change the neighbor parameter with (-r/-k)')
        plt.show()
        sys.exit()

    elif(rmax<= min(radius+dx,tmp)) :
        print('ERROR: There is a problem with the r_max detection, it is shorter than the largest distance in the neighbors graph.')
        print('       You may consider to fix the first detection of r_max with option -M and/or the r_min with option -np to fix the fit range' )
        print('       or to decrease the neighbors parameter with (-r/-k)')
        plt.show()
        sys.exit()
    #1

    #2 Finding actual r_max and std. dev. to define fitting interval [rmin;rM] 
    distr_x=h[1][0:n_bins]+dx/2 # the actual distance r
    distr_y=h[0][0:n_bins] # the frequency p(r) [probability]

    res= np.empty(data_shape[1])
    left_distr_x = np.empty(n_bins)
    left_distr_y = np.empty(n_bins)
    left_distr_x= distr_x[np.logical_and(distr_x[:]>rmax-std, distr_x[:]<rmax+std/2.0)]
    left_distr_y= np.log(distr_y[np.logical_and(distr_x[:]>rmax-std, distr_x[:]<rmax+std/2.0)])
    coeff = np.polyfit(left_distr_x,left_distr_y,2,full='False')    
    a0=coeff[0][0]
    b0=coeff[0][1]
    c0=coeff[0][2]

    rmax = -b0/a0/2.0
    if(args.r_max>0) : rmax=args.r_max 
    std=np.sqrt(abs(-1/a0/2.))

    left_distr_x= distr_x[np.logical_and(distr_x[:]>rmax-std, distr_x[:]<rmax+std/2.)]
    left_distr_y= np.log(distr_y[np.logical_and(distr_x[:]>rmax-std, distr_x[:]<rmax+std/2.)])
    coeff = np.polyfit(left_distr_x,left_distr_y,2,full='False')
    a=coeff[0][0]
    b=coeff[0][1]
    c=coeff[0][2]

    rmax_old=rmax
    std_old=std
    rmax = abs(-b/a/2.)
    std=np.sqrt(abs(-1/a/2.))   # it was a0
    rmin=max(rmax-2*np.sqrt(abs(-1/a/2.))-dx/2,0.)
    if(args.r_min>=0) : 
        rmin=args.r_min
    elif (rmin < radius and args.r_min!=-1) : 
        rmin = radius 
        print('\nWARNING: For internal consistency r_min has been fixed to the largest distance (r_NN_max) in the neighbors graph.')
        print('         It is possible to reset the standard definition of r_min=r_max-2*sigma running with option "-np -1" ' )
        print('         or you can use -np to manually define a desired value (Example: -np 0.1)\np' )
      
    rM=rmax+dx/4

    if(np.fabs(rmax-rmax_old)>std_old/4 ) :    #fit consistency check
        print('\nWARNING: The histogram is probably not smooth enough (you may try to change n_bin with -b), rmax is fixed to the value of first iteration\np'  )
        #print rmax,rmax_old,std/4,std_old/4
        rmax=rmax_old
        a=a0
        b=b0
        c=c0
        if(args.r_min>=0) :
            rmin=args.r_min
        elif (rmin < radius and args.r_min!=-1) :
            rmin = radius
            print('\nWARNING2: For internal consistency r_min has been fixed to the largest distance in the neighbors graph (r_NN_max).')
            print('          It is possible to reset the standard definition of r_min=r_max-2*sigma running with option "-np -1" ')
            print('          or you can use -np to manually define a desired value (Example: -np 0.1)\np')
        rM=rmax+dx/4
    #2

    #3 Gaussian Fitting to determine ratio R
    left_distr_x= distr_x[np.logical_and(np.logical_and(distr_x[:]>rmin,distr_x[:]<=rM),distr_y[:]>0.000001)]/rmax
    left_distr_y= np.log(distr_y[np.logical_and(np.logical_and(distr_x[:]>rmin,distr_x[:]<=rM),distr_y[:]>0.000001)])-(4*a*c-b**2)/4./a
    
    fit =  curve_fit(func2,left_distr_x,left_distr_y)
    ratio=np.sqrt(fit[0][0])
    y1=func2(left_distr_x,fit[0][0])
    #3

    #4 Geodesics D-Hypersphere Distribution Fitting to determine Dfit
    fit = curve_fit(func,left_distr_x,left_distr_y)
    Dfit=(fit[0][0])+1


    y2=func(left_distr_x,fit[0][0],fit[0][1],fit[0][2])
    #4


    #5 Determination of Dmin
    D_file = open('{}_D_residual.dat'.format(filename), "w")

    for D in range(1,data_shape[1]+1):
        y=(func(left_distr_x,D-1,1,0))
        for i in range(0, len(y)):
            res[D-1] = np.linalg.norm((y)-(left_distr_y))/np.sqrt(len(y))
        D_file.write("%s " % D)
        D_file.write("%s\np" % res[D-1])

    Dmin = np.argmax(-res)+1

    D_file.close()

    y=func(left_distr_x,Dmin-1,fit[0][1],0)
    #5

    #6 Printing results
    print('\nFITTING PARAMETERS:' )
    print('rmax, std. dev., rmin', rmax,std,rmin)
    print('\nFITTING RESULTS:' )
    print('R, Dfit, Dmin', ratio,Dfit,Dmin , '\np')

    if(Dmin == 1) : print('NOTE: Dmin = 1 could indicate that the choice of the input parameters is not optimal or simply an underestimation of a 2D manifold\np')
    fit_file= open('{}_fit.dat'.format(filename), "w")

    for i in range(0, len(y)):
        fit_file.write("%s " % left_distr_x[i])
        fit_file.write("%s " % ((left_distr_y[i])))
        fit_file.write("%s " % ((y1[i])))
        fit_file.write("%s " % ((y2[i])))
        fit_file.write("%s\np" % ((y[i])))
    fit_file.close() 

         
    stat_file= open('{}_statistics.dat'.format(filename), "w")
    statistics = str('# Npoints, rmax, standard deviation, R, D_fit, Dmin \np# \
    {}, {}, {}, {}, {}, {}\np'.format(num_connect,rmax,std,ratio,Dfit,Dmin))
    stat_file.write("%s" % statistics)
    for i in range(0, len(distr_x)-2): 
        stat_file.write("%s " % distr_x[i])
        stat_file.write("%s " % distr_y[i])
        stat_file.write("%s\np" % np.log(distr_y[i]))
    stat_file.close()

    plt.figure(2)
    plt.plot(left_distr_x,left_distr_y,'o-',label='Representation (K='+str(args.n_neighbors)+')')
    plt.plot(left_distr_x,y1,label='Gaussian (m={})'.format(int(Dmin)))
    plt.plot(left_distr_x,y2,label='Hypersphere (m={})'.format(int(Dmin)))
    # plt.plot(left_distr_x,y,label='Hypersphere (m$_{min}$)')
    plt.xlabel(r'$log (r/r_{max})$', fontproperties=axis_label_prop)
    plt.ylabel(r'$log (p(r)/p(r_{max}))$', fontproperties=axis_label_prop)
    plt.legend(loc=4, prop=font_prop)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.grid(True)
    plt.savefig(filename+'_fit.pdf', bbox_inches = "tight")  


    plt.figure(3)
    plt.plot(range(1,data_shape[1]+1),res,'o-',label='m (K='+str(args.n_neighbors)+')')
    plt.legend(prop=font_prop)
    plt.xlabel('Dimension', fontproperties=axis_label_prop)
    plt.ylabel('Root Mean Squared Error', fontproperties=axis_label_prop)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.grid(True)
    plt.savefig(filename+'_Dmin.pdf', bbox_inches = "tight")


    #6

if __name__ == '__main__':
    main()