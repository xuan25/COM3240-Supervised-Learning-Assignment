import subprocess
import os

notification = False

base_cmd = [ "python", "main.py" ]

if not os.path.exists('./log/baseline/'):
    os.makedirs('./log/baseline/')
subprocess.run(base_cmd + [ "20", "50", "0.05", "0.001", "0", "0", "0", './log/baseline/{}.csv'.format(0) ])

if not os.path.exists('./log/l1/'):
    os.makedirs('./log/l1/')
for lam in [ "0", "1.0e-9", "1.0e-7", "1.0e-5", "1.0e-4", "1.0e-3", "5.0e-5", "5.0e-4", "3.0e-4" ]:
    subprocess.run(base_cmd + [ "500", "50", "0.05", "0.001", lam, "50", "0", './log/l1/{}.csv'.format(lam) ])

if not os.path.exists('./log/num_neuron/'):
    os.makedirs('./log/num_neuron/')
for n_neuron in [ 10, 50, 100, 150, 250, 400, 800  ]:
    for i in range(10):
        subprocess.run(base_cmd + [ "250", "50", "0.05", "0.001", "1.0e-4", str(n_neuron), '0', './log/num_neuron/{}.{}.csv'.format(n_neuron, i) ])

if not os.path.exists('./log/deep_layer/'):
    os.makedirs('./log/deep_layer/')
for n_neuron in [ 10, 50, 100, 150, 250, 400, 800 ]:
    for i in range(10):
        subprocess.run(base_cmd + [ "250", "50", "0.05", "0.001", "1.0e-4", "100", str(n_neuron), './log/deep_layer/{}.{}.csv'.format(n_neuron, i) ])

if notification:
    from playsound import playsound
    playsound("C:\Windows\Media\Windows Foreground.wav")
