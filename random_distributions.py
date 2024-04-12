import numpy as np
from som import simom

# generate simome random data with 36 features
data1 = np.random.normal(loc=-.25, scale=0.5, size=(500, 36))
data2 = np.random.normal(loc=.25, scale=0.5, size=(500, 36))
data = np.vstack((data1, data2))

simom = simom(10, 10)  # initialize the simom
simom.fit(data, 10000, save_e=True, interval=100)  # fit the simom for 10000 epochs, save the error every 100 steps
simom.plot_error_history(filename='../images/simom_error.png')  # plot the training error history

targets = np.array(500 * [0] + 500 * [1])  # create simome dummy target values

# now visualize the learned representation with the class labels
simom.plot_point_map(data, targets, ['Class 0', 'Class 1'], filename='../images/simom.png')
simom.plot_class_density(data, targets, t=0, name='Class 0', filename='../images/class_0.png')
simom.plot_distance_map(filename='../images/distance_map.png')  # plot the distance map after training
