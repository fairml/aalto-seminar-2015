import sys

from itertools import chain, combinations, product
from collections import defaultdict
from optparse import OptionParser
import operator
import copy
from apriori import runApriori
from evaluation import *
from algorithms import *


def printResults(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    for item, support in sorted(items, key=lambda (item, support): support):
        print "item: %s , %.3f" % (str(item), support)
    print "\n------------------------ RULES:"
    for rule, confidence in sorted(rules.items(), key=operator.itemgetter(1)):    
        pre, post = rule
        print "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)

def binarize(r):
    filtered_r = set()
    for j in r:
        i = j.split(': ')
        if i[0] == "age":
            if int(i[1]) > 30:
                filtered_r.add(i[0] + ': ' + '>30')
            else:
                filtered_r.add(i[0] + ': ' + '<=30')
        elif i[0] == "workclass":
            if i[1] in ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"]:
                filtered_r.add(i[0] + ': ' + 'Job')
            else:
                filtered_r.add(i[0] + ': ' + 'No-Job')
        elif i[0] == "fnlwgt":
            if int(i[1]) > 100000:
                filtered_r.add(i[0] + ': ' + '>100000')
            else:
                filtered_r.add(i[0] + ': ' + '<=100000')
        elif i[0] == "education":
            if i[1] in ["Bachelors", "Masters", "Doctorate"]:
                filtered_r.add(i[0] + ': ' + 'Degree')
            else:
                filtered_r.add(i[0] + ': ' + 'No-Degree')
        elif i[0] == "education-num":
            if int(i[1]) > 10:
                filtered_r.add(i[0] + ': ' + '>10')
            else:
                filtered_r.add(i[0] + ': ' + '<=10')
        elif i[0] == "marital-status":
            if i[1] in ["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"]:
                filtered_r.add(i[0] + ': ' + 'Married')
            else:
                filtered_r.add(i[0] + ': ' + 'Not-Married')
        elif i[0] == "occupation":
            if i[1] in ["Protective-serv", "Armed-Forces"]:
                filtered_r.add(i[0] + ': ' + 'Army')
            else:
                filtered_r.add(i[0] + ': ' + 'Not-Army')
        elif i[0] == "relationship":
            if i[1] in ["Wife", "Husband"]:
                filtered_r.add(i[0] + ': ' + 'Couple')
            else:
                filtered_r.add(i[0] + ': ' + 'No-Couple')
        elif i[0] == "race":
            if i[1] in ["White"]:
                filtered_r.add(i[0] + ': ' + 'White')
            else:
                filtered_r.add(i[0] + ': ' + 'Not-White')
        elif i[0] == "sex":
            filtered_r.add(i[0] + ': ' + i[1])
        elif i[0] == "capital-gain":
            if int(i[1]) > 1000:
                filtered_r.add(i[0] + ': ' + '>1000')
            else:
                filtered_r.add(i[0] + ': ' + '<=1000')
        elif i[0] == "capital-loss":
            if int(i[1]) > 1000:
                filtered_r.add(i[0] + ': ' + '>1000')
            else:
                filtered_r.add(i[0] + ': ' + '<=1000')
        elif i[0] == "hours-per-week":
            if int(i[1]) > 40:
                filtered_r.add(i[0] + ': ' + '>40')
            else:
                filtered_r.add(i[0] + ': ' + '<=40')
        elif i[0] == "native-country":
            if i[1] in ["United-States"]:
                filtered_r.add(i[0] + ': ' + 'United-States')
            else:
                filtered_r.add(i[0] + ': ' + 'Not-United-States')
        else:
            filtered_r.add(j)
    return frozenset(filtered_r)
        
def semi_binarize(r):
    filtered_r = set()
    for j in r:
        i = j.split(': ')
        if i[0] == "age":
            if int(i[1]) > 30:
                filtered_r.add(i[0] + ': ' + '>30')
            else:
                filtered_r.add(i[0] + ': ' + '<=30')
        else:
            filtered_r.add(j)        
    return frozenset(filtered_r)


def dataFromFile(fname, mode='original'):
    """Function which reads from the file and yields a generator"""
    fields_all = "age: workclass: fnlwgt: education: education-num: marital-status: occupation: relationship: race: sex: capital-gain: capital-loss: hours-per-week: native-country: label:"
    fields = fields_all.split(' ')
    file_iter = open(fname, 'rU')
    for line in file_iter:
        line = line.strip().rstrip(', ')  # Remove trailing comma
        record = frozenset([i[0] + ' ' + i[1] for i in zip(fields, line.split(', '))])
        if mode == 'bin':
            record = binarize(record)
        elif mode == 's-bin':
            record = semi_binarize(record)

        yield record
        
def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList


if __name__ == "__main__":

    alpha = 1.2
    p = 0.8

    mode = 'original' # original dataset
    #mode = 'bin' # binarized dataset
    #mode = 's-bin' # semi-binarized dataset
    inFile = dataFromFile('adult.txt', mode=mode)

    itemSet, transactionList = getItemSetTransactionList(inFile)

    minSupport, minConfidence = 0.09, 0.6

    print 'Apriori is running'
    items, rules, freqSet = runApriori(itemSet, transactionList, minSupport, minConfidence)
    printResults(items, rules)
    print 'number of frequent itemsets:', len(items)
    print 'number of frequent association rules:', len(rules)

    DI_s = frozenset(["sex: Female", "marital-status: Never-married"])
    #DI_s = frozenset(["sex: Female", "age: <=30"])
    #DI_s = frozenset(["marital-status: Not-Married", "education: No-Degree"])

    print 'discriminatory itemset:',DI_s

    MRs, PRs = get_MRs(rules, alpha, DI_s)
    print 'num of alpha-discriminatory rules', len(MRs)
    print 'num of alpha-protective rules', len(PRs)
    Rs, NRs = get_PRs(rules, freqSet, alpha, DI_s)
    print 'num of redlining rules and non-redlining', len(Rs)
    print 'num of non-redlining rules', len(NRs)

    #run algorithm 1:
    to_add, to_remove = alg1(transactionList, rules, freqSet, alpha, DI_s)

    #run algorithm 2:
    #to_add, to_remove = alg2(transactionList, rules, freqSet, alpha, DI_s)

    #run algorithm 3:
    #to_add, to_remove = alg3(transactionList, rules, freqSet, alpha, DI_s, p = 0.8, mode = "Method2")

    #run algorithm 4:
    #to_add, to_remove = alg4(transactionList, rules, freqSet, alpha, DI_s, p = 0.8)

    # construct new dataset:
    new_transactionList = list(set(transactionList).difference(set(to_remove)).union(set(to_add)))

    #run Baseline:
    #new_transactionList = filter_DI(transactionList, sensitive=['marital-status', 'education'])

    print 'Apriori is running again'
    new_items, new_rules, new_freqSet = runApriori(itemSet, new_transactionList, minSupport, minConfidence)
    print '---------- statistics of transformed dataset ---------------'
    print 'number of frequent itemsets:', len(new_items)
    print 'number of frequent association rules:', len(new_rules)

    new_MRs, new_PRs = get_MRs(new_rules, alpha, DI_s)
    print 'num of alpha-discriminatory rules', len(new_MRs)
    print 'num of alpha-protective rules', len(new_PRs)
    new_Rs, new_NRs = get_PRs(new_rules, new_freqSet, alpha, DI_s)
    print 'num of redlining rules and non-redlining', len(new_Rs)
    print 'num of non-redlining rules', len(new_NRs)

    print 'DDPD:', 1.0*len(set(MRs).difference(set(new_MRs)))/len(MRs)
    print 'DDPP:', 1.0*len(set(PRs).intersection(set(new_PRs)))/len(PRs)
    print 'IDPD:', 1.0*len(set(Rs).difference(set(new_Rs)))/len(Rs)
    print 'IDPP:', 1.0*len(set(NRs).intersection(set(new_NRs)))/len(NRs)
    print 'MC:', 1.0*len(set((rules.keys())).difference(set(new_rules.keys())))/len(rules.keys())
    print 'GC:', 1.0*len(set((new_rules.keys())).difference(set(rules.keys())))/len(new_rules.keys())