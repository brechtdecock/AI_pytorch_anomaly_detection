# -*- coding: utf-8 -*-
"""
An example for SVDD model fitting with negataive samples
example 1 uses unlabeled date: X values are x/y coordinates, there is no Y(which are the labels)
"""
import numpy as np
import base_SVDD

# create 100 points with 2 dimensions
n = 100
dim = 2
X = np.r_[np.random.randn(n, dim)]

# svdd object using rbf kernel
svdd = base_SVDD.BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')

# fit the SVDD model
svdd.fit(X)

# predict the label
y_predict = svdd.predict(X)

# plot the boundary
svdd.plot_boundary(X)

# plot the distance
radius = svdd.radius
distance = svdd.get_distance(X)
svdd.plot_distance(radius, distance)