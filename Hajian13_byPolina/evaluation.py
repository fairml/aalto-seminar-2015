import algorithms

def filter_DI(transactionList, sensitive=['age', 'sex']):
    new_transactionList = []
    for r in transactionList:
        filtered_r = set()
        for j in r:
            i = j.split(': ')
            if i[0] not in sensitive:
                filtered_r.add(j)
        new_transactionList.append(filtered_r)

    return new_transactionList

def get_MRs(freq_rules, alpha, DI_s):
    MRs, PRs = [], []
    for r, conf in freq_rules.iteritems():
        if algorithms.is_MR(freq_rules, DI_s, r, alpha):
            MRs.append(r)
        else:
            PRs.append(r)
    return MRs, PRs

def get_PRs(freq_rules, freqSet, alpha, DI_s):
    Rs, NRs = [], []
    for r, conf in freq_rules.iteritems():
        test = algorithms.is_RR(r, freq_rules, freqSet, alpha, DI_s)
        if test[0]:
            Rs.append(r)
        else:
            NRs.append(r)
    return Rs, NRs