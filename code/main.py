import math
import sys
import time

from scipy.io import loadmat
import numpy as np      # CPU computing 
# import cupy as np       # GPU computing 



# -------- Common --------

trace_train_err = True
trace_train_acc = True
trace_weight_updates = True
trace_valid_err = True
trace_valid_acc = True
enable_test = True

export_csv = True
csv_file = 'trace.csv'

notification = False

start_time = time.time()

init_with_commandline = len(sys.argv) > 1



# -------- Hyperparameters --------

n_epoch = 100       # number of epochs to train
batch_size = 50     # batch size
eta = 0.05          # eta is the learning rate (Excessive learning rate may cause np.inf and thus np.nan)
tau = 0.001         # tau for exponential moving average
l1_lam = 1.0e-4     # lambda for L1 regularisation penalty

n_hidden_layer1 = 100       # number of neurons of the hidden layer1. 0 deletes this layer
n_hidden_layer2 = 0         # number of neurons of the hidden layer2. 0 deletes this layer

a_leaky = 0.01      # alpha for Leaky ReLU [0,1]

# -------- Override form commandline --------
# python main.py <n_epoch> <batch_size> <eta> <tau> <l1_lam> <n_hidden_layer1> <n_hidden_layer2> <csv_file>

if init_with_commandline:
    n_epoch = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    eta = float(sys.argv[3])
    tau = float(sys.argv[4])
    l1_lam = float(sys.argv[5])

    n_hidden_layer1 = int(sys.argv[6])
    n_hidden_layer2 = int(sys.argv[7])

    print('Hyperparameters overridden by commandline.')

    csv_file = sys.argv[8]



# -------- Logging to csv --------

if export_csv:
    import csv
    csvfile = open(csv_file, 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['epoch', 'train_error', 'train_accuracy', 'average_weight_update', 'valid_error', 'valid_accuracy', 'test_accuracy'])
    csvfile.flush()



# -------- Load data --------

# The MNIST contains the char A to Z so we will set the number of labels as 26
n_labels = 26

# Load data from emnist
emnist = loadmat('emnist-letters-1k.mat')
emnist_train_images = np.array(emnist['train_images'])
emnist_train_labels = np.array(emnist['train_labels'])
emnist_test_images = np.array(emnist['test_images'])
emnist_test_labels = np.array(emnist['test_labels'])

emnist_train_labels = emnist_train_labels.flatten()
emnist_test_labels = emnist_test_labels.flatten()

# Input normalization
emnist_train_images = emnist_train_images / 255
emnist_test_images = emnist_test_images / 255

# Shuffle emnist index
emnist_train_index = np.random.permutation(emnist_train_images.shape[0])
emnist_test_index = np.random.permutation(emnist_test_images.shape[0])

# Index for train/valid/test
emnist_train_p80 = int((emnist_train_index.shape[0] + emnist_test_index.shape[0]) * 0.8 - emnist_test_index.shape[0])
train_index = emnist_train_index[:emnist_train_p80]
valid_index = emnist_train_index[emnist_train_p80:]
test_index = emnist_test_index

# Init datasets
train_image = emnist_train_images[train_index]
train_label = emnist_train_labels[train_index]
valid_image = emnist_train_images[valid_index]
valid_label = emnist_train_labels[valid_index]
test_image = emnist_test_images
test_label = emnist_test_labels

# Init desired outputs
train_y = np.zeros((train_label.shape[0], n_labels))
for i in range(0,train_label.shape[0]):    
    train_y[i, train_label[i].astype(int)] = 1

valid_y = np.zeros((valid_label.shape[0], n_labels))
for i in range(0,valid_label.shape[0]):    
    valid_y[i, valid_label[i].astype(int)] = 1

test_y = np.zeros((test_label.shape[0], n_labels))
for i in range(0,test_label.shape[0]):    
    test_y[i, test_label[i].astype(int)] = 1

# Data set attributes
img_size = emnist_train_images.shape[1]
n_train = train_index.shape[0]
n_valid = valid_index.shape[0]
n_test = test_index.shape[0]

# Print shape
print("train: {}".format(n_train))
print("valid: {}".format(n_valid))
print("test: {}".format(n_test))



# -------- Init network --------
# He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers:
# Surpassing human-level performance on imagenet classification. 
# In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034). 
# Retrieved from https://arxiv.org/abs/1502.01852

# define the size of input & output layers 
n_input_layer = img_size
n_output_layer = n_labels

# Initialize network parameters
if n_hidden_layer1 > 0:
    W1 = np.random.randn(n_input_layer, n_hidden_layer1) * np.sqrt(2 / n_input_layer)
    if n_hidden_layer2 > 0:
        W2 = np.random.randn(n_hidden_layer1, n_hidden_layer2) * np.sqrt(2 / n_hidden_layer1)
        W3 = np.random.randn(n_hidden_layer2, n_output_layer) * np.sqrt(2 / n_hidden_layer2)
    else:
        W2 = np.random.randn(n_hidden_layer1, n_output_layer) * np.sqrt(2 / n_hidden_layer1)
else:
    W1 = np.random.randn(n_input_layer, n_output_layer) * np.sqrt(2 / n_input_layer)

# Initialize the biases
if n_hidden_layer1 > 0:
    W1_bias = np.zeros((n_hidden_layer1,))
    if n_hidden_layer2 > 0:   
        W2_bias=np.zeros((n_hidden_layer2,))
        W3_bias=np.zeros((n_output_layer,))
    else:
        W2_bias = np.zeros((n_output_layer,))
else:
    W1_bias = np.zeros((n_output_layer,))



# -------- Init activation function --------
# Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013, June). 
# Rectifier nonlinearities improve neural network acoustic models. 
# In Proc. icml (Vol. 30, No. 1, p. 3).
# Retrieved from https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf

# activation function
def activation_func(x):
    return x * (x > 0) + a_leaky * x * (x <= 0)     # Leaky ReLU
    # return x * (x > 0)                              # Faster ReLU
    # return np.maximum(x, 0)                         # ReLU
    # return 1.0/(1.0+np.exp(-x))                     # Sigmoid

# gradient of activation function 
def grad_activation_func(x):
    return 1 * (x > 0) + a_leaky * (x <= 0)                 # Leaky ReLU grad
    # return 1 * (x > 0)                                      # ReLU grad
    # return activation_func(x)*(1.0-activation_func(x))      # Sigmoid



# -------- Init L1 regularisation penalty function --------

# L1 regularisation penalty
# w_list_all is a list of all the weight matrix in the network
# return a scalar
def l1_penalty(lambda_1, w_list_all):
    sum_abs = 0
    for w in w_list_all:
        sum_abs += np.sum(np.abs(w))
    l1_pen = lambda_1 * sum_abs
    return l1_pen

# gradient of L1 regularisation penalty
# w is a weight matrix
# return a matrix
def grad_l1_penalty(lambda_1, w):
    grad_l1_pen = w / np.abs(w)
    grad_l1_pen[np.isnan(grad_l1_pen)] = 0
    grad_l1_pen = lambda_1 * grad_l1_pen
    return grad_l1_pen
        


# -------- Init performance trackers --------

train_errors = np.zeros((n_epoch,))
train_accuracies = np.zeros((n_epoch,))
valid_errors = np.zeros((n_epoch,))
valid_accuracies = np.zeros((n_epoch,))
average_weight_update = None
weight_updates = np.zeros((n_epoch,))



# -------- Start Training --------

last_time = time.time()

# number of batches
n_batches = int(math.ceil(n_train / batch_size))


for epoch in range(0,n_epoch):
    # Train for a epoch

    # Shuffle the order of the samples each epoch
    shuffled_idxs = np.random.permutation(n_train)
    shuffled_train_images = train_image[shuffled_idxs]
    shuffled_desired_output = train_y[shuffled_idxs]
    
    for batch in range(0,n_batches):
        # Train for a batch
        x0 = np.array(shuffled_train_images[batch*batch_size : (batch+1)*batch_size])
        desired_output = np.array(shuffled_desired_output[batch*batch_size : (batch+1)*batch_size])

        # ---- Forward propagation ----
        last_layer_output = x0
        if True:
            h1 = np.dot(last_layer_output, W1) + W1_bias
            x1 = activation_func(h1)

            last_layer_output = x1

        if n_hidden_layer1 > 0:
            h2 = np.dot(last_layer_output, W2) + W2_bias
            x2 = activation_func(h2)

            last_layer_output = x2
        
        if n_hidden_layer2 > 0:
            h3 = np.dot(last_layer_output, W3) + W3_bias
            x3 = activation_func(h3)

            last_layer_output = x3

        # ---- Backpropagation ----
        e_n = last_layer_output - desired_output
        e_for_next_layer = e_n

        if n_hidden_layer2 > 0:
            delta3 = grad_activation_func(h3) * e_for_next_layer
            dW3 = (np.sum(delta3[:,:,np.newaxis] * x2[:,np.newaxis,:], axis=0) / batch_size).T  # https://stackoverflow.com/questions/16500426/is-there-a-more-vectorized-way-to-perform-numpy-outer-along-an-axis
            dW3_bias = np.sum(delta3, axis=0) / batch_size 
            
            e_for_next_layer = np.dot(delta3, W3.T)
        
        if n_hidden_layer1 > 0:
            delta2 = grad_activation_func(h2) * e_for_next_layer
            dW2 = (np.sum(delta2[:,:,np.newaxis] * x1[:,np.newaxis,:], axis=0) / batch_size).T
            dW2_bias = np.sum(delta2, axis=0) / batch_size 

            e_for_next_layer = np.dot(delta2, W2.T)

        if True:
            delta1 = grad_activation_func(h1) * e_for_next_layer
            dW1 = (np.sum(delta1[:,:,np.newaxis] * x0[:,np.newaxis,:], axis=0) / batch_size).T            
            dW1_bias = np.sum(delta1, axis=0) / batch_size 

            # e_for_next_layer = np.dot(delta1, W1.T)   # final (not required)

        # ---- Weight update ----
        if True:
            l1_pen_w1 = grad_l1_penalty(l1_lam, W1)
            l1_pen_w1_bias = grad_l1_penalty(l1_lam, W1_bias)
            DW1 = -eta * (dW1 + l1_pen_w1)
            DW1_bias = -eta * (dW1_bias + l1_pen_w1_bias)
            W1 += DW1
            W1_bias += DW1_bias
        
        if n_hidden_layer1 > 0:
            l1_pen_w2 = grad_l1_penalty(l1_lam, W2)
            l1_pen_w2_bias = grad_l1_penalty(l1_lam, W2_bias)
            DW2 = -eta * (dW2 + l1_pen_w2)
            DW2_bias = -eta * (dW2_bias + l1_pen_w2_bias)
            W2 += DW2
            W2_bias += DW2_bias

        if n_hidden_layer2 > 0:
            l1_pen_w3 = grad_l1_penalty(l1_lam, W3)
            l1_pen_w3_bias = grad_l1_penalty(l1_lam, W3_bias)
            DW3 = -eta * (dW3 + l1_pen_w3)
            DW3_bias = -eta * (dW3_bias + l1_pen_w3_bias)
            W3 += DW3
            W3_bias += DW3_bias

        # ---- Trace performance ----
        if trace_train_err:
            # MSE error
            mse = 0.5 * np.sum(np.square(e_n)) / n_train

            # L1 penalty
            w_list = []
            if True:
                w_list.extend([W1, W1_bias])
            if n_hidden_layer1 > 0:
                w_list.extend([W2, W2_bias])
            if n_hidden_layer2 > 0:
                w_list.extend([W3, W3_bias])
            l1_pen = l1_penalty(l1_lam, w_list)
            
            # Store the error per epoch
            train_errors[epoch] = train_errors[epoch] + mse + l1_pen
        else:
            train_errors[epoch] = -1

        if trace_train_acc:
            # Training accuracy
            output_label = np.argmax(last_layer_output, axis=1)
            desired_label = np.argmax(desired_output, axis=1)
            corr_count = np.sum(output_label == desired_label)
            accuracy = 100 * corr_count / n_train
            train_accuracies[epoch] = accuracy
        else:
            train_accuracies[epoch] = -1

        # After each batch update (init) the exponential moving average matrix
        if trace_weight_updates:
            weight_deltas = []
            if True:
                weight_deltas.append(DW1.flatten())
                weight_deltas.append(DW1_bias.flatten())
            
            if n_hidden_layer1 > 0:
                weight_deltas.append(DW2.flatten())
                weight_deltas.append(DW2_bias.flatten())

            if n_hidden_layer2 > 0:
                weight_deltas.append(DW3.flatten())
                weight_deltas.append(DW3_bias.flatten())

            # Store all weight_deltas into a flatten tensor
            weight_deltas = np.concatenate(weight_deltas)

            if average_weight_update is None:
                average_weight_update = weight_deltas
            else:
                average_weight_update = (1 - tau) * average_weight_update + tau * weight_deltas

    # ---- End of an batch ----
    if trace_weight_updates:
        weight_updates[epoch] = np.sum(average_weight_update)
    else:
        weight_updates[epoch] = -1

    # ---- Validation ----
    if trace_valid_err or trace_valid_acc:
        x0 = valid_image

        last_layer_output = x0
        if True:
            h1 = np.dot(last_layer_output, W1) + W1_bias
            x1 = activation_func(h1)

            last_layer_output = x1

        if n_hidden_layer1 > 0:
            h2 = np.dot(last_layer_output, W2) + W2_bias
            x2 = activation_func(h2)

            last_layer_output = x2
        
        if n_hidden_layer2 > 0:
            h3 = np.dot(last_layer_output, W3) + W3_bias
            x3 = activation_func(h3)

            last_layer_output = x3

        if trace_valid_err:
            # Validation error
            e_n = last_layer_output - valid_y
            mse = 0.5 * np.sum(np.square(e_n)) / n_valid

            # L1 penalty
            w_list = []
            if True:
                w_list.extend([W1, W1_bias])
            if n_hidden_layer1 > 0:
                w_list.extend([W2, W2_bias])
            if n_hidden_layer2 > 0:
                w_list.extend([W3, W3_bias])
            l1_pen = l1_penalty(l1_lam, w_list)
            
            # Store the errors
            valid_errors[epoch] = mse + l1_pen
        else:
            valid_errors[epoch] = -1

        if trace_valid_acc:
            # Validation accuracy
            output_label = np.argmax(last_layer_output, axis=1)
            corr_count = np.sum(output_label == valid_label)
            accuracy = 100 * corr_count / n_valid
            valid_accuracies[epoch] = accuracy
        else:
            valid_accuracies[epoch] = -1
    else:
        valid_errors[epoch] = -1
        valid_accuracies[epoch] = -1

    # ---- End of an epoch ----
    curr_time = time.time()
    epoch_dura = curr_time - last_time
    last_time = curr_time

    print( "[{} ({} s)] Epoch {}: t_err={}, t_acc%={}, w_update={}, v_err={}, v_acc%={}".format(time.strftime("%d %b %Y %H:%M:%S", time.localtime(curr_time)), epoch_dura, epoch + 1, train_errors[epoch], train_accuracies[epoch], weight_updates[epoch], valid_errors[epoch], valid_accuracies[epoch]))
    if export_csv:
        csvwriter.writerow([epoch + 1, train_errors[epoch], train_accuracies[epoch], weight_updates[epoch], valid_errors[epoch], valid_accuracies[epoch], None])
        csvfile.flush()

# ---- End of the training ----
end_time = time.time()
total_dura = end_time - start_time
print( "Total duration: {} s".format(total_dura))



# -------- Test accuracy --------
if enable_test:
    x0 = test_image
    last_layer_output = x0

    if True:
        h1 = np.dot(last_layer_output, W1) + W1_bias
        x1 = activation_func(h1)

        last_layer_output = x1

    if n_hidden_layer1 > 0:
        h2 = np.dot(last_layer_output, W2) + W2_bias
        x2 = activation_func(h2)

        last_layer_output = x2

    if n_hidden_layer2 > 0:
        h3 = np.dot(last_layer_output, W3) + W3_bias
        x3 = activation_func(h3)

        last_layer_output = x3

    output_label = np.argmax(last_layer_output, axis=1)

    corr_count = np.sum(output_label == test_label)
    accuracy = 100 * corr_count / n_test 
    print("Accuracy on test = {} %".format(accuracy))
    if export_csv:
        csvwriter.writerow([None, None, None, None, None, None, accuracy])
        csvfile.flush()



# -------- Finished --------

print("Finished")

if notification:
    from playsound import playsound
    playsound("C:\Windows\Media\Windows Foreground.wav")
