# Assignment 1: Supervised learning of handwritten letters

## Quick start guide

1. Place the dataset file `emnist-letters-1k.mat` in the root of the project directory.
2. Execute the command `python main.py` in the shell to start the training.

## Run with specified Hyperparameters

Hyperparameters can be configured with two methods.

### Method 1: Set in the python script

1. Open script `main.py` with an editor.
2. Find the section `Hyperparameters`.
    - You should see content similar to
```python
# -------- Hyperparameters  --------

n_epoch = 100       # number of epochs to train
batch_size = 50     # batch size
eta = 0.05          # eta is the learning rate (Excessive learning rate may cause np.inf and thus np.nan)
tau = 0.001         # tau for exponential moving average
l1_lam = 1.0e-4     # lambda for L1 regularisation penalty

n_hidden_layer1 = 100       # number of neurons of the hidden layer1. 0 deletes this layer
n_hidden_layer2 = 0         # number of neurons of the hidden layer2. 0 deletes this layer
```
3. Edit hyperparameters
4. Execute the command `python main.py` in the shell to start the training.

### Method 2: Set with command-line

Execute the command `python main.py <n_epoch> <batch_size> <eta> <tau> <l1_lam> <n_hidden_layer1> <n_hidden_layer2> <csv_file>` in the shell to start the training.

e.g.: `python main.py 250 50 0.05 0.001 1.0e-4 100 0 trace.csv`

where the parameter `<csv_file>` is the location for the output of the performance trace log, and the other parameters have the same naming as the hyperparameters in the script.

Note: Set parameters with command-line will override those in the script.

<div style='page-break-after: always;'></div>

## Additional Features

The section `Common` in the `main.py` allows you to configure additional features of the script, where each parameter has the following meaning.

 - trace_train_err: Whether training errors are traced during the training process
 - trace_train_acc: Whether training accuracy are traced during the training process
 - trace_weight_updates: Whether Average Weight Updates are traced during the training process
 - trace_valid_err: Whether validation errors are traced during the training process
 - trace_valid_acc: Whether validation accuracy are traced during the training process
 - enable_test: Whether the test accuracy is calculated at the end of the training
 - export_csv: Whether to export the trace log to a csv file
 - csv_file: Location of trace log file output
 - notification: Whether to play a beep at the end of the program (For Windows)

## Auxiliary Scripts

### job.py

This script is used to automatically execute all the training tasks required in the report in batch and generate csv trace logs.

### mean_std.py

This script is used to calculate the mean and standard deviation of the test accuracy from log files, and print it to the screen.
