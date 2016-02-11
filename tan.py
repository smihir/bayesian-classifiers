from __future__ import division
import copy
import sys
import itertools
import math

class Vertex:
    def __init__(self, id):
        self.id = id
        self.neighbors = dict()

    def add_neighbor(self, v, weight = 0):
        self.neighbors[v] = weight

    def get_edge__weight(self, v):
        return self.neighbors[v]

class Graph:
    def __init__(self):
        self.adjacency_list = dict()
        # vlist is for debugging purposes only
        self.vlist = list()

    def add_vertex(self, id):
        v = Vertex(id)
        self.adjacency_list[v] = list()

        self.vlist.append(v)

    def get_vertex_by_id(self, id):
        for v in self.adjacency_list:
            if v.id == id:
                return v
        return None

    def add_edge_by_id(self, id1, id2, cost = 0):
        v1 = self.get_vertex_by_id(id1)
        v2 = self.get_vertex_by_id(id2)

        if not v1 or not v2:
            raise NameError('VertexNotFound')

        self.adjacency_list[v1].append(v2)
        self.adjacency_list[v2].append(v1)

        v1.add_neighbor(v2, cost)
        v2.add_neighbor(v1, cost)

class Tan:
    def __init__(self, fname):
        self.arff = __import__("arff")
        self.data = None
        self.raw_test_data = None
        self.test_data = None
        self.model = dict()
        self.attribute_dictionary = dict()
        self.bayes_net = Graph()
        with open(fname) as f:
            self.raw_data = self.arff.load(f)

        self.data = copy.deepcopy(self.process_raw_data(self.raw_data['data']))
        self.make_attribute_dictionary()

        self.create_bayes_net()

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

    def create_bayes_net(self):
        # add vertices to graph, dont add the classifier
        classifier = self.raw_data['attributes'][-1][0]
        for attr in self.attribute_dictionary.keys():
            if attr == classifier:
                continue
            self.bayes_net.add_vertex(attr)

        for v1, v2 in itertools.combinations(self.bayes_net.adjacency_list.keys(), 2):
            attr1 = v1.id
            attr2 = v2.id
            mutual_information = 0
            for classval in self.attribute_dictionary[classifier]:
                for attrval1, attrval2 in itertools.product(self.attribute_dictionary[attr1],
                                                            self.attribute_dictionary[attr2]):
                    #print(attr1, attrval1, attr2, attrval2, classval)
                    jp2 = self.conditional_probabilty2(attr1, attrval1, attr2, attrval2, classval, classifier)
                    jpattr1 = self.conditional_probability(attr1, attrval1, classval, classifier)
                    jpattr2 = self.conditional_probability(attr2, attrval2, classval, classifier)
                    jp3 = self.joint_probability3(attr1, attrval1, attr2, attrval2, classifier, classval)
                    #print(jp2, jpattr1, jpattr2, jp3, jp2/(jpattr1*jpattr2))
                    mutual_information += jp3 * math.log(jp2/(jpattr1*jpattr2), 2)
            #print(attr1, attr2, mutual_information)
            self.bayes_net.add_edge_by_id(attr1, attr2, mutual_information)


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

    def conditional_probabilty2(self, fname1, fval1, fname2, fval2, classifierval, classifier = 'class',
                                laplace = True):
        total = len([1 for d in self.data if d[classifier] == classifierval])
        conditional_count = len([1 for d in self.data if d[fname1] == fval1 and d[fname2] == fval2 and
                                 d[classifier] == classifierval])

        if laplace:
            conditional_count += 1
            total += len(self.attribute_dictionary[fname1]) * len(self.attribute_dictionary[fname2])

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

    def joint_probability3(self, fname1, fval1, fname2, fval2, fname3, fval3, laplace = True):
        total = len(self.data)
        count = len([1 for d in self.data if d[fname1] == fval1 and d[fname2] == fval2 and d[fname3] == fval3])

        if laplace:
            count += 1
            total += len(self.attribute_dictionary[fname1]) * len(self.attribute_dictionary[fname2]) * \
                     len(self.attribute_dictionary[fname3])

        return count / total


if __name__ == "__main__":
    t = Tan(sys.argv[1])
    print("Done")