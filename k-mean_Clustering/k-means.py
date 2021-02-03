#!/usr/bin/python

# This program attend to read data from a csv file,
# and apply kmean, then output the result.

from pylab            import plot,show
from numpy            import vstack,array
from numpy.random     import rand
from scipy.cluster.vq import kmeans, vq, whiten

import csv

if __name__ == "__main__":

    # clusters
    K = 2

    data_arr = []
    ages_arr = []

    with open('tshirts-J.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data_arr.append([float(x) for x in row[1:]])
            ages_arr.append([row[0]])

    data = vstack( data_arr )
    meal_name = vstack(ages_arr)

    # normalization
   # data = whiten(data)

centroids, distortion = kmeans(data,2)

idx,_ = vq(data,centroids)

print(centroids)
print (idx)
print (data_arr)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()
    