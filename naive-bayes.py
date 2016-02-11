from __future__ import division
import copy
import sys
import operator

class NaiveBayes:
    """This is the Naive Bayes classifier

    The constructor takes name if the training file as the argument.
    This class does lazy evaluation, so only training data is loaded
    and no calculations are done. Calculation is done for each entry
    in test data.

    Classification begins with the call to classify() method with
    the test arff file as the input argument.
    """
    def __init__(self, fname):
        self.arff = __import__("arff")
        self.data = None
        self.raw_test_data = None
        self.test_data = None
        self.model = dict()
        self.attribute_dictionary = dict()
        with open(fname) as f:
            self.raw_data = self.arff.load(f)

        self.data = copy.deepcopy(self.process_raw_data(self.raw_data['data']))
        self.make_attribute_dictionary()

    def make_attribute_dictionary(self):
        for attr in self.raw_data['attributes']:
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
        with open(testf) as f:
            self.raw_test_data = self.arff.load(f)
        self.test_data = copy.deepcopy(self.process_raw_data(self.raw_test_data['data']))

        classifier = self.raw_data['attributes'][-1][0]
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
            print(mval[0] + ' ' + td[classifier] + str(" %.12f" %(mval[1] / tval)))
        print("\n" + str(correct))

if __name__ == "__main__":
    nb = NaiveBayes(sys.argv[1])
    nb.classify(sys.argv[2])