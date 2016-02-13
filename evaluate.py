from __future__  import division
import matplotlib.pyplot as plt
from naivebayes import NaiveBayes
from tan import Tan
import numpy as np
import sys

def evaluate_tan():
    t = Tan(sys.argv[1], evaluate = True)
    out = t.evaluate(sys.argv[2])
    process(out, 'TAN')

def evaluate_naivebayes():
    nb = NaiveBayes(sys.argv[1], evaluate = True)
    out = nb.evaluate(sys.argv[2])
    process(out, 'Naive Bayes')

def process(out, classifier):
    x = list()
    y = list()

    for run in out:
        test_data_size = run[0][1]
        train_data_size = run[0][2]
        c = [d[0] for d in run]
        avg_correct = sum(d[0] for d in run) / len(run)
        x.append(train_data_size)
        y.append(avg_correct / test_data_size)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Accuracy vs. Training Data Size for ' + classifier)
    ax.set_xlabel('Training Data Size')
    ax.set_ylabel('Accuracy')
    ax.plot(x, y, 'ro')
    ax.plot(x, y, c='b')

def plot():
    plt.show()

if __name__ == '__main__':
    evaluate_tan()
    evaluate_naivebayes()
    plot()
