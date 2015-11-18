import sys

from itertools import chain, combinations, product
from collections import defaultdict
from optparse import OptionParser
import operator
import copy


def is_MR(rules, DI_s, r, alpha):

    if r[0].intersection(DI_s) and r[0].difference(DI_s) and 1.0 * rules[r] / rules[(r[0].difference(DI_s), r[1])] >= alpha:
        return True
    return False

def get_PN(frequent_rules, DI_s, B, C, alpha):
    out = {}
    for r, c in frequent_rules.iteritems():
        if not is_MR(frequent_rules, DI_s, r, alpha) and (r[0].union(r[1])).issuperset(B.union(C)) and (r[0].union(r[1])) != B.union(C):
            out[r] = c
    return out


def is_RR(r, freq_rules, freqSet, alpha, DI_s):
    # assume that is_MR check was done before
    X, C = list(r[0]), r[1]
    minConfidence = 0.7
    found, RRs = False, []

    mask = ["".join(seq) for seq in product("01", repeat=len(r[0]))]
    mask.remove("1" * len(r[0]))
    mask.remove("0" * len(r[0]))
    for m in mask:
        D, B = set(), set()
        for i in xrange(len(m)):
            bit = m[i]
            if bit == '1':
                D.add(X[i])
            else:
                B.add(X[i])

        for A in DI_s:
            a = set()
            a.add(A)
            sup_ab = freqSet[frozenset(a.union(B))]
            sub_db = freqSet[frozenset(D.union(B))]
            if sup_ab > 0 and sub_db > 0:
                abd_conf = 1.0*freqSet[frozenset(a.union(B).union(D))]/sup_ab
                dba_conf = 1.0*freqSet[frozenset(D.union(B).union(a))]/sub_db
                mr = (frozenset(a.union(B)), frozenset(C))
                if abd_conf >= minConfidence and dba_conf >= minConfidence and mr in freq_rules and is_MR(freq_rules, DI_s, mr, alpha):
                    found = True
                    RRs.append((frozenset(a), frozenset(B), frozenset(C), frozenset(D)))
    return (found, RRs)


def get_space(transactionList):
    possible = {}
    for i in transactionList:
        for j in i:
            item = j.split(' ')
            k, v = item[0][:-1], item[1]
            if k not in possible:
                possible[k] = set()
            possible[k].add(v)
    return possible


def get_negations(possible, itemset):
    out = []
    itemset_list = list(itemset)

    mask = ["".join(seq) for seq in product("01", repeat=len(itemset))]
    mask.remove("1" * len(itemset))
    for m in mask:
        tmp = []
        for i in xrange(len(m)):
            if m[i] == '1':
                tmp_list = [itemset_list[i]]
            else:
                tmp_list = []
                item = itemset_list[i].split(' ')
                k, v = item[0][:-1], item[1]
                t = set()
                t.add(v)
                for j in possible[k].difference(t):
                    tmp_list.append(k + ': ' + j)
            tmp.append(tmp_list)

        negs = list(product(*tmp))
        for i in negs:
            out.append(frozenset(i))
    return out

def alg5(freq_rules, freqSet, alpha, DI_s, p = 0.8):
    TR = {}
    rb = {}
    for r_1, conf_1 in freq_rules.iteritems():
        if is_MR(freq_rules, DI_s, r_1, alpha):
            A = (r_1[0]).intersection(DI_s)
            B = r_1[0].difference(A)
            C = r_1[1]
            D_pn = get_PN(freq_rules, DI_s, B, C, alpha)
            if D_pn:
                stats = {}
                counter1, counter2 = 0,0
                mindif, rule_mindif = sys.maxint, (-1,-1)
                nt = False
                for r, conf in D_pn.iteritems():
                    D = r[0].difference(B)
                    conf_2 = 1.0*freqSet[A.union(B).union(D)]/freqSet[A.union(B)]
                    if (conf >= p*conf_1):
                        c1r = True
                        dif1 = 0.0
                        counter1 += 1
                    else:
                        c1r = False
                        dif1 = p*conf_1 - conf
                        if mindif > dif1:
                            mindif = dif1
                            rule_mindif = (A,B,C,D)
                    if conf_2 >= p:
                        c2r = True
                        counter2 += 1
                    else:
                        c2r = False
                    if c2r and c1r:
                        nt = True
                        break
                    #stats[r] = (c1r, c2r, dif1)

                if nt:
                    TR[r_1] = 'Nt'
                elif counter2 == 0:
                    TR[r_1] = 'DRP'
                else:
                    TR[r_1] = 'RG'
                    rb[r_1] = rule_mindif
            else:
                TR[r_1] = 'DRP'

            if TR[r_1] == 'RG':
                dif_prime = conf_1 - alpha*1.0*freqSet[B.union(C)]/freqSet[B]
                if dif_prime < mindif:
                    TR[r_1] = 'DRP'


    return TR, rb

def alg2_4_9(possible, r, transactionList, freq_rules, freqSet, alpha, DI_s):
    to_add, to_remove = [], []

    A = (r[0]).intersection(DI_s)
    B = r[0].difference(A)
    C = r[1]

    negList = []
    negA = get_negations(possible, A)
    negC = get_negations(possible, C)
    for i in negA:
        negList.append(i.union(B).union(negC[0]))
    supRecs = []

    for i in transactionList:
        for j in negList:
            if i.issuperset(j):
                supRecs.append(i)
    impacts = {}
    for i in supRecs:
        impacts[i] = 0.0
        for j in freq_rules.keys():
            if i.issuperset(j[0]):
                impacts[i] += 1.0

    sorted_supRecs = sorted(impacts.items(), key=operator.itemgetter(1))

    counter = 0
    sup_abc = freqSet[r[0].union(r[1])]
    sup_ab = freqSet[r[0]]

    sup_bc = freqSet[B.union(r[1])]
    sup_b = freqSet[B]

    while counter < len(sorted_supRecs) and 1.0*sup_abc / sup_ab >= alpha * 1.0*sup_bc/sup_b:
        to_remove.append(sorted_supRecs[counter][0])
        sup_bc += 1
        tmp = copy.deepcopy(sorted_supRecs[counter][0])
        tmp = tmp.difference(negC[0])
        to_add.append(tmp.union(r[1]))
        counter += 1

    return to_add, to_remove

def alg1_5_14(possible, r, transactionList, freq_rules, freqSet, alpha, DI_s):
    to_add, to_remove = [], []

    A = (r[0]).intersection(DI_s)
    B = r[0].difference(A)
    C = r[1]

    negList = []
    negA = get_negations(possible, A)
    negC = get_negations(possible, C)
    for i in negA:
         negList.append(i.union(B).union(negC[0]))
    supRecs = []

    for i in transactionList:
        for j in negList:
            if i.issuperset(j):
                supRecs.append(i)
    impacts = {}

    for i in supRecs:
        impacts[i] = 0.0
        for j in freq_rules.keys():
            if i.issuperset(j[0]):
                impacts[i] += 1.0

    sorted_supRecs = sorted(impacts.items(), key=operator.itemgetter(1))
    counter = 0
    sup_abc = freqSet[r[0].union(r[1])]
    sup_ab = freqSet[r[0]]

    while counter < len(sorted_supRecs) and 1.0*sup_abc / sup_ab >= alpha * freq_rules[(B, r[1])]:
        to_remove.append(sorted_supRecs[counter][0])
        sup_ab += 1
        tmp = copy.deepcopy(sorted_supRecs[counter][0])
        for i in negA:
            tmp = tmp.difference(i)
        to_add.append(tmp.union(A))
        counter += 1

    return to_add, to_remove

def alg4(transactionList, freq_rules, freqSet, alpha, DI_s, p = 0.8):
    to_add, to_remove = [], []
    possible = get_space(transactionList)
    for r, conf in freq_rules.iteritems():
        X = r[0]
        if not is_MR(freq_rules, DI_s, r, alpha):
            testRR = is_RR(r, freq_rules, freqSet, alpha, DI_s)
            if testRR[0]:
                for RR in testRR[1]:

                    A, B, C, D = RR[0], RR[1], RR[2], RR[3]
                    b2 = 1.0*freqSet[X.union(A)]/freqSet[X]
                    d = 1.0*freqSet[B.union(C)]/freqSet[B]
                    b1 = 1.0*freqSet[X.union(A)]/freqSet[B.union(A)]

                    negA = get_negations(possible, A)
                    negD = get_negations(possible, D)
                    negC = get_negations(possible, C)

                    negList = []
                    for na in negA:
                        for nd in negD:
                            negList.append(na.union(B).union(nd).union(negC[0]))
                    supRecs = []
                    for i in transactionList:
                        for j in negList:
                            if i.issuperset(j):
                                supRecs.append(i)
                    impacts = {}
                    for i in supRecs:
                        impacts[i] = 0.0
                        for j in freq_rules.keys():
                            if i.issuperset(j[0]):
                                impacts[i] += 1.0

                    sorted_supRecs = sorted(impacts.items(), key=operator.itemgetter(1))

                    if is_MR(freq_rules, DI_s, (A.union(B),C), alpha):
                        sup_bc = freqSet[B.union(C)]
                        sup_b = freqSet[B]
                        counter = 0
                        conf1 = 1.0*freqSet[A.union(B).union(C)]/freqSet[A.union(B)]

                        while counter < len(sorted_supRecs) and 0.1*p*sup_bc/sup_b <= b1*(b2+conf-1)/(b2*alpha) and 0.1*p*sup_bc/sup_b <= conf1/alpha:
                            to_remove.append(sorted_supRecs[counter][0])
                            sup_bc += 1.0
                            tmp = copy.deepcopy(sorted_supRecs[counter][0])
                            tmp = tmp.difference(negC[0])
                            to_add.append(tmp.union(C))
                            counter += 1
                    else:
                        sup_bc = freqSet[B.union(C)]
                        sup_b = freqSet[B]
                        counter = 0
                        while counter < len(sorted_supRecs) and 0.1*p*sup_bc/sup_b <= b1*(b2+conf-1)/(b2*alpha):
                            to_remove.append(sorted_supRecs[counter][0])
                            sup_bc += 1.0
                            tmp = copy.deepcopy(sorted_supRecs[counter][0])
                            tmp = tmp.difference(negC[0])
                            to_add.append(tmp.union(C))
                            counter += 1

    return to_add, to_remove


def alg3(transactionList, freq_rules, freqSet, alpha, DI_s, p = 0.8, mode = "Method1"):
    TR, rb = alg5(freq_rules, freqSet, alpha, DI_s, p)
    possible = get_space(transactionList)
    to_remove = []
    to_add = []
    for r, stat in TR.iteritems():
        if stat == 'RG':
            A, B, C, D = rb[r]
            negD = get_negations(possible, D)
            negC = get_negations(possible, C)

            negList = []
            for i in negD:
                negList.append(i.union(A).union(B).union(C))
            supRecs = []
            for i in transactionList:
                for j in negList:
                    if i.issuperset(j):
                        supRecs.append(i)
            impacts = {}
            for i in supRecs:
                impacts[i] = 0.0
                for j in freq_rules.keys():
                    if i.issuperset(j[0]):
                        impacts[i] += 1.0

            sorted_supRecs = sorted(impacts.items(), key=operator.itemgetter(1))
            counter = 0
            sup_abc = freqSet[r[0].union(r[1])]
            sup_ab = freqSet[r[0]]
            sup_dbc = freqSet[D.union(B).union(C)]
            sup_db = freqSet[D.union(B)]

            while counter < len(sorted_supRecs) and 0.1*p*sup_abc / sup_ab > 1.0*sup_dbc/sup_db:
                to_remove.append(sorted_supRecs[counter][0])
                sup_abc -= 1.0
                tmp = copy.deepcopy(sorted_supRecs[counter][0])
                tmp = tmp.difference(C)
                to_add.append(tmp.union(negC[0]))
                counter += 1

        elif stat == 'DRP':
            if mode == 'Method1':

                add, remove = alg1_5_14(possible, r, transactionList, freq_rules, freqSet, alpha, DI_s)
                to_add += add
                to_remove += remove
            else:
                add, remove = alg2_4_9(possible, r, transactionList, freq_rules, freqSet, alpha, DI_s)
                to_add += add
                to_remove += remove

    return to_add, to_remove

def alg2(transactionList, freq_rules, freqSet, alpha, DI_s):
    possible = get_space(transactionList)

    to_remove = []
    to_add = []
    for r, conf in freq_rules.iteritems():
        if is_MR(freq_rules, DI_s, r, alpha):
            A = (r[0]).intersection(DI_s)
            B = r[0].difference(A)
            negA = get_negations(possible, A)
            negC = get_negations(possible, r[1])

            negList = []
            for i in negA:
                negList.append(i.union(B).union(negC[0]))
            supRecs = []

            for i in transactionList:
                for j in negList:
                    if i.issuperset(j):
                        supRecs.append(i)
            impacts = {}
            for i in supRecs:
                impacts[i] = 0.0
                for j in freq_rules.keys():
                    if i.issuperset(j[0]):
                        impacts[i] += 1.0

            sorted_supRecs = sorted(impacts.items(), key=operator.itemgetter(1))

            counter = 0
            sup_abc = freqSet[r[0].union(r[1])]
            sup_ab = freqSet[r[0]]

            sup_bc = freqSet[B.union(r[1])]
            sup_b = freqSet[B]

            while counter < len(sorted_supRecs) and 1.0*conf >= alpha * 1.0*sup_bc/sup_b:
                to_remove.append(sorted_supRecs[counter][0])
                sup_bc += 1
                tmp = copy.deepcopy(sorted_supRecs[counter][0])
                tmp = tmp.difference(negC[0])
                to_add.append(tmp.union(r[1]))
                counter += 1

    return to_add, to_remove

def alg1(transactionList, freq_rules, freqSet, alpha, DI_s):
    possible = get_space(transactionList)

    to_remove = []
    to_add = []
    for r, conf in freq_rules.iteritems():
        if is_MR(freq_rules, DI_s, r, alpha):
            A = (r[0]).intersection(DI_s)
            B = r[0].difference(A)
            negA = get_negations(possible, A)
            negC = get_negations(possible, r[1])

            negList = []
            for i in negA:
                negList.append(i.union(B).union(negC[0]))
            supRecs = []

            for i in transactionList:
                for j in negList:
                    if i.issuperset(j):
                        supRecs.append(i)
            impacts = {}
            for i in supRecs:
                impacts[i] = 0.0
                for j in freq_rules.keys():
                    if i.issuperset(j[0]):
                        impacts[i] += 1.0

            sorted_supRecs = sorted(impacts.items(), key=operator.itemgetter(1))
            counter = 0
            sup_abc = freqSet[r[0].union(r[1])]
            sup_ab = freqSet[r[0]]

            while counter < len(sorted_supRecs) and 1.0*sup_abc / sup_ab >= alpha * freq_rules[(B, r[1])]:
                to_remove.append(sorted_supRecs[counter][0])
                sup_ab += 1
                tmp = copy.deepcopy(sorted_supRecs[counter][0])
                for i in negA:
                    tmp = tmp.difference(i)
                to_add.append(tmp.union(A))
                counter += 1
    return to_add, to_remove
