import matplotlib.pyplot as plt

training_fraction_list = [ 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
logistic_accuracy = [0.823, 0.842, 0.73, 0.74, 0.72, 0.64]
naive_accuracy = [0.85, 0.865, 0.874, 0.873, 0.874, 0.879]


plt.plot(training_fraction_list, naive_accuracy, 'r', label = "naive")
plt.plot(training_fraction_list, logistic_accuracy, 'g', label = "logistic-regression")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()