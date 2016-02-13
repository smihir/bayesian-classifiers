from __future__ import division
import copy
import sys
import itertools
import math
import operator
import random

class Vertex:
    def __init__(self, id):
        self.id = id
        self.neighbors = dict()
        # parents are used for quick lookups in directed graphs
        self.parents = list()

    def add_neighbor(self, v, weight = 0):
        self.neighbors[v] = weight

    def add_parent(self, p):
        self.parents.append(p)

    def get_neighbor_weight(self, v):
        if self.neighbors.has_key(v):
            return self.neighbors[v]
        else:
            return None

    def is_connected(self, v):
        if self.neighbors.has_key(v):
            return True
        else:
            return False

class Graph:
    def __init__(self, directed = False):
        self.adjacency_list = list()
        # vlist is for debugging purposes only
        self.directed = directed

    def add_vertex(self, id):
        v = Vertex(id)
        self.adjacency_list.append(v)

    def get_vertex_by_id(self, id):
        for v in self.adjacency_list:
            if v.id == id:
                return v
        return None

    def get_edge_weight(self, frm, to):
        return frm.get_neighbor_weight(to)

    def add_edge_by_id(self, frm, to, cost = 0):
        v1 = self.get_vertex_by_id(frm)
        v2 = self.get_vertex_by_id(to)

        if not v1 or not v2:
            raise NameError('VertexNotFound')

        v1.add_neighbor(v2, cost)

        if not self.directed:
            v2.add_neighbor(v1, cost)
        else:
            v2.add_parent(v1)

class Tan:
    def __init__(self, fname, evaluate = False):
        self.arff = __import__("arff")
        self.data = None
        self.raw_test_data = None
        self.test_data = None
        self.attribute_dictionary = dict()
        self.bayes_net = Graph()
        self.spanning_tree = None
        self.attribute_no_lookup = dict()
        with open(fname) as f:
            self.eval_data = self.arff.load(f)

        i = 0
        for v in self.eval_data['attributes']:
            self.attribute_no_lookup[v[0]] = i
            i += 1
        self.make_attribute_dictionary()

        if not evaluate:
            self.raw_data = copy.deepcopy(self.eval_data)
            self.generate_model()

    def generate_model(self):
        self.data = copy.deepcopy(self.process_raw_data(self.raw_data['data']))
        self.create_bayes_net()
        self.find_maximum_spanning_tree()

    def clean_training_data(self):
        self.data = None
        self.raw_test_data = None
        self.test_data = None
        self.bayes_net = Graph()
        self.spanning_tree = None

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

    def create_bayes_net(self):
        # add vertices to graph, dont add the classifier
        classifier = self.raw_data['attributes'][-1][0]
        for attr in self.attribute_dictionary.keys():
            if attr == classifier:
                continue
            self.bayes_net.add_vertex(attr)

        for v1, v2 in itertools.combinations(self.bayes_net.adjacency_list, 2):
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

    def find_maximum_spanning_tree(self):
        self.spanning_tree = Graph(directed = True)
        Q = list()
        for v in self.raw_data['attributes']:
            if v[0] == self.raw_data['attributes'][-1][0]:
                continue
            Q.append(v[0])
        self.spanning_tree.add_vertex(Q.pop(0))

        while True:
            if len(Q) == 0:
                break
            max_weight = -1
            next_vertex = None
            for vertex in self.spanning_tree.adjacency_list:
                for q_vertex in Q:
                    bn_vertex_a = self.bayes_net.get_vertex_by_id(vertex.id)
                    bn_vertex_q = self.bayes_net.get_vertex_by_id(q_vertex)
                    val = bn_vertex_a.get_neighbor_weight(bn_vertex_q)
                    if val > max_weight:
                        max_weight = val
                        next_vertex = q_vertex
                        parent_vertex = vertex
            self.spanning_tree.add_vertex(next_vertex)
            self.spanning_tree.add_edge_by_id(parent_vertex.id, next_vertex, max_weight)
            Q.remove(next_vertex)
            #print(self.attribute_no_lookup[parent_vertex.id], self.attribute_no_lookup[next_vertex.id])


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

    def conditional_probability_by2(self, fname1, fval1, pname, pval, classifierval, classifier = 'class',
                                    laplace = True):
        total = len([1 for d in self.data if d[classifier] == classifierval and d[pname] == pval])
        conditional_count = len([1 for d in self.data if d[classifier] == classifierval and d[pname] == pval and
                                d[fname1] == fval1])

        if laplace:
            conditional_count += 1
            total += len(self.attribute_dictionary[fname1])

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

    def classify(self, testf):
        classifier = self.raw_data['attributes'][-1][0]

        for v in self.raw_data['attributes']:
            if v[0] == classifier:
                continue
            if len(self.spanning_tree.get_vertex_by_id(v[0]).parents) == 0:
                print(v[0] + " " + classifier)
            else:
                parentkey = self.spanning_tree.get_vertex_by_id(v[0]).parents[0].id
                print(v[0] + " " + parentkey + " " + classifier)
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
                    if len(self.spanning_tree.get_vertex_by_id(key).parents) == 0:
                        cp *= self.conditional_probability(key, value, ckey, classifier)
                    else:
                        parentkey = self.spanning_tree.get_vertex_by_id(key).parents[0].id
                        parentval = td[parentkey]
                        cp *= self.conditional_probability_by2(key, value, parentkey, parentval, ckey, classifier)

                cplist[ckey] = cp

            tval = sum(map(lambda (x, y): y, cplist.iteritems()))
            mval = max(cplist.iteritems(), key=operator.itemgetter(1))

            if mval[0] == td[classifier]:
                correct += 1
            print(mval[0] + ' ' + td[classifier] + str(" %.12f" %(mval[1] / tval)))
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
    t = Tan(sys.argv[1])
    t.classify(sys.argv[2])