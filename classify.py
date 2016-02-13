from naivebayes import NaiveBayes
from tan import Tan
import sys

def tan(trainf, testf):
    t = Tan(trainf)
    t.classify(testf)

def naivebayes(trainf, testf):
    nb = NaiveBayes(trainf)
    nb.classify(testf)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Not enough arguments')
    else:
        if (sys.argv[3] == 'n'):
            naivebayes(sys.argv[1], sys.argv[2])
        if (sys.argv[3] == 't'):
            tan(sys.argv[1], sys.argv[2])
