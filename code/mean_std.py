import csv
import numpy as np

print('num_neuron')
print('x,y,std')
for n_neuron in [ 10, 50, 100, 150, 250, 400, 800 ]:
    test_accs = []
    for i in range(0, 10):
        f_name = './log/num_neuron/{}.{}.csv'.format(n_neuron, i)
        with open(f_name,'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            testrow = rows[-1]
            test_acc = testrow[6]
            test_accs.append(test_acc)
    test_accs = np.array(test_accs, dtype=np.float)
    mean = np.mean(test_accs)
    std = np.std(test_accs)
    print('{},{},{}'.format(n_neuron, mean, std))
print()

print('deep_layer')
print('x,y,std')
for n_neuron in [ 10, 50, 100, 150, 250, 400, 800 ]:
    test_accs = []
    for i in range(0, 10):
        f_name = './log/deep_layer/{}.{}.csv'.format(n_neuron, i)
        with open(f_name,'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            testrow = rows[-1]
            test_acc = testrow[6]
            test_accs.append(test_acc)
    test_accs = np.array(test_accs, dtype=np.float)
    mean = np.mean(test_accs)
    std = np.std(test_accs)
    print('{},{},{}'.format(n_neuron, mean, std))
print()
