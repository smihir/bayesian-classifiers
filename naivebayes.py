from __future__ import division
import copy
import sys
import operator
import random

class NaiveBayes:
    """This is the Naive Bayes classifier

    The constructor takes name if the training file as the argument.
    This class does lazy evaluation, so only training data is loaded
    and no calculations are done. Calculation is done for each entry
    in test data.

    Classification begins with the call to classify() method with
    the test arff file as the input argument.
    """
    def __init__(self, fname, evaluate = False):
        self.arff = __import__("arff")
        self.data = None
        self.raw_test_data = None
        self.test_data = None
        self.attribute_dictionary = dict()
        with open(fname) as f:
            self.eval_data = self.arff.load(f)
        self.make_attribute_dictionary()

        if not evaluate:
            self.raw_data = copy.deepcopy(self.eval_data)
            self.generate_model()

    def generate_model(self):
        self.data = copy.deepcopy(self.process_raw_data(self.raw_data['data']))

    def clean_training_data(self):
        self.data = None
        self.raw_test_data = None
        self.test_data = None

    def make_attribute_dictionary(self):
        for attr in self.eval_data['attributes']:
            self.attribute_dictionary[attr[0]] = attr[1]

    def process_raw_data(self, raw_data):
        res = list()
        for rd in raw_data:
            di = dict()
            i = 0
            for key in self.raw_data['attributes']:
                di[key[0]] = rd[i]
                i += 1

            res.append(di)
        return res

    def conditional_probability(self, fname, fval, classifierval, classifier = 'class', laplace = True):
        """
        :param fname: Name of the feature
        :param fval: Value of the feature
        :param classifierval: Value of the feature which has to be classified
        :param classifier: Name of the feature which has to be classified
        :param laplace: Use Laplace estimates?
        :return: Conditional Probability
            P(fname = fval | classifier = classifierval)
        """
        total = len([1 for d in self.data if d[classifier] == classifierval])
        conditional_count = len([1 for d in self.data if d[classifier] == classifierval and d[fname] == fval])

        if laplace:
            conditional_count += 1
            total += len(self.attribute_dictionary[fname])

        return conditional_count / total

    def probability(self, fname, fval, laplace = True):
        """
        :param fname: Name of the feature
        :param fval: Value of the feature
        :param laplace: Use Laplace estimates?
        :return: Probability
            P(fname = fval)
        """
        total = len(self.data)
        count = sum(1 for d in self.data if d[fname] == fval)

        if laplace:
            total += len(self.attribute_dictionary[fname])
            count += 1

        return count / total

    def classify(self, testf):
        classifier = self.raw_data['attributes'][-1][0]
        for v in self.raw_data['attributes']:
            if v[0] == classifier:
                continue
            print(v[0] + " " + classifier)
        print("")

        with open(testf) as f:
            self.raw_test_data = self.arff.load(f)
        self.test_data = copy.deepcopy(self.process_raw_data(self.raw_test_data['data']))

        p = dict()
        correct = 0
        for v in self.raw_data['attributes'][-1][1]:
            p[v] = self.probability(classifier, v)

        for td in self.test_data:
            cplist = dict()
            for ckey, cval in p.iteritems():
                cp = cval
                for key, value in td.iteritems():
                    if key == classifier:
                        continue
                    cp *= self.conditional_probability(key, value, ckey, classifier)

                cplist[ckey] = cp

            tval = sum(map(lambda (x, y): y, cplist.iteritems()))
            mval = max(cplist.iteritems(), key=operator.itemgetter(1))

            if mval[0] == td[classifier]:
                correct += 1
            print(mval[0] + ' ' + td[classifier] + str(" %.12f" %(mval[1] / tval)).rstrip('0'))
        print("\n" + str(correct))
        return correct, len(self.raw_test_data['data'])

    def evaluate(self, tname):
        random.seed(100)
        #ratios = [0.25, 0.5, 1]
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = list()

        self.raw_data = dict()
        self.raw_data['attributes'] = self.eval_data['attributes']
        for r in ratios:
            run = list()
            for c in range(1, 5):
                rindex = random.sample(range(0, len(self.eval_data['data'])), int(r * len(self.eval_data['data'])))
                self.clean_training_data()
                self.raw_data['data'] = list()
                for i in rindex:
                    self.raw_data['data'].append(self.eval_data['data'][i])
                self.generate_model()
                correct, total = self.classify(tname)
                run.append((correct, total, int(r * len(self.eval_data['data']))))
            result.append(run)
        return result

if __name__ == "__main__":
    nb = NaiveBayes(sys.argv[1])
    nb.classify(sys.argv[2])
