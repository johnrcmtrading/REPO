It is my intellectual property, if you want to use it you are bound to let me know the purpose (simoneambrogio80@gmail.com).


# Qiskit-Perceptron-iris-DB-set-
It binary classifies after shuffling, without training.


# REQUIREMENTS
I used IMB Qiskit Library for Python 3.6.


# PURPOSE
In this repo you can find the Iris dataframe problem. A simple problem, but faced by only manipulating a 3-qubits circuit.
I commented as much I can and the aim of the algo is to classify each sample inputed (X[row]) in an unsupervised binary fashion.


# STRUCTURE
'Qperceptron_iriscomm' is the main file.
Library 'DB_iris.py' creates/distributes the classes and 'Qperceptron_iriscomm.py' tells you 'same class' or 'different class' after measurement over 3 qubits. The manipulation of the qubits is my personal variation of QFT algos.


# PARAMETERS
Only relevant parameters are:

- AGAINST_CLASS_1 (T/F, Iris classes are 3, thereore it asks you if you want class 0 to be checked and classified against class 1 or against class 2)

- divisorpower [NO NEED TO CHANGE IT] (it is a function of the number of columns [features] in the dataframe... ~0.13 x the number of columns... no need to play with it in this stage)

- REAL_DEVICE (True if you want to run on ibm real quantum machine... I use a token.txt after registration in https://quantum-computing.ibm.com/).


# PRINTOUT
In the end it prints the accuracy. I used 2 models ('straight' that leverages the function qft, and 'reverse' that leverages the function qft_rev).
The best of the 2 models never hits for this problem something > 7% classification error (you will see ERROR SCORES: [[2.] [4.]] for example, meaning that the first model's error is 2%, the second model's error is 4%).
Note that the names 'straight', 'reverse' to the models are for identification purpose only. There is no straight and reverse indeed.

