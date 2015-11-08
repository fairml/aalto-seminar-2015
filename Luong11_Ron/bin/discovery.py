#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.tree as skt


def ordinal_conv(value):
    # ------------German Credit Ordinal Mapping Dictionary---------------
    ordinal_d = dict()
    rank = np.array([0, 1, 2, 3, 4]).astype(np.float)
    ordinal_d = {'A11': rank[0] / rank[2],
               'A12': rank[1] / rank[2],
               'A13': rank[2] / rank[2],
               'A14': np.nan,
               'A61': rank[0] / rank[3],
               'A62': rank[1] / rank[3],
               'A63': rank[2] / rank[3],
               'A64': rank[3] / rank[3],
               'A65': np.nan,
               'A71': np.nan,
               'A72': rank[0] / rank[3],
               'A73': rank[1] / rank[3],
               'A74': rank[2] / rank[3],
               'A75': rank[3] / rank[3],
               }
    # -------------------------------------------------------------------

    conversion = ordinal_d[value]

    return conversion


def parse_data(filename, col_header, delimiter, continuous, discrete, decision):
    # Delimiter: True - White Space; False - Comma
    data = pd.read_csv(filename,
                       delim_whitespace=delimiter,
                       names=col_header,
                       skipinitialspace=True)

    # -------------------Z-Score and Ordinal Mapping---------------------
    mean = []
    sigma = []
    for column in discrete + continuous:

        # Calculate Z-score
        if column in continuous:

            # Window of Analysis
            interval = data.ix[:, column]

            # Variable Mean
            m = np.mean(interval)
            numerator = interval.values - m*np.ones(len(interval))

            # Mean Absolute Deviation
            s = sum(abs(numerator))/float(len(numerator))

            # Z-score
            z = numerator/s

            # Assignments
            data.loc[:, column] = z
            mean.append(m)
            sigma.append(s)

        # Convert Ordinals to intervals
        if column in discrete:
            for row in range(len(data)):
                data.loc[row, column] = ordinal_conv(data[column][row])

    mean = pd.Series(mean, index=continuous)
    sigma = pd.Series(sigma, index=continuous)
    # -------------------------------------------------------------------

    # Retrieve the Labels
    label_class = data.loc[:, decision]

    # Remove the label from the dataset
    data.drop(decision, axis=1, inplace=True)

    return data, label_class, mean, sigma


def discriminate(protected, data):

    # --------------Evaluate Which Attributes are Protected --------------
    g = []
    for param in range(len(protected)):

        # Protected
        if protected[param] is not None:
            g.append(param)

            # List of possible values for the attribute
            if len(protected[param]) > 1 and isinstance(protected[param], list):

                # Interval (Lower and Upper Bounds)
                if param in int_var:
                    protected[param][0] = (protected[param][0] - mu[param])/sigma[param]
                    protected[param][1] = (protected[param][1] - mu[param])/sigma[param]

            # Convert Ordinal Variable
            if param in ord_var:
                protected[param] = ordinal_conv(protected[param])
    # -------------------------------------------------------------------

    res = pd.notnull(data.loc[:, g])
    for feature in g:

        # List of values for the Attribute
        if isinstance(protected[feature], list):

            # Numerical Boundaries
            if feature in int_var + ord_var:
                lb = data[feature] >= protected[feature][0]
                ub = data[feature] <= protected[feature][1]
                res.loc[:, feature] = lb.eq(ub)
            else:
                # Nominal alternatives
                for row in data.index:

                    # Test individual entries against the protected set
                    if data.loc[row, feature] in protected[feature]:
                        res.loc[row, feature] = True
                    else:
                        res.loc[row, feature] = False
        else:
            # Point comparison (Nominal, continuous or discrete variable)
            res.loc[:, feature] = data[feature] == protected[feature]

    # Divide the samples in protected and unprotected groups
    prot = []
    unprot = []
    for sample in res.index:

        # Protected satisfies the True attribute for all protected variables
        if (res.loc[sample, :] == True).all():
            prot.append(sample)
        else:
            unprot.append(sample)

    return prot, unprot


def rank(s_idx, source, target, protected, num_l, ord_l, nom_l):

    # Identify non-protected attributes
    g = []
    for i in range(len(protected)):
        if protected[i] is not None:
            if i in num_l:
                num_l.remove(i)
            elif i in ord_l:
                ord_l.remove(i)
            elif i in nom_l:
                nom_l.remove(i)
        else:
            g.append(i)

    numerical = num_l + ord_l

    # Pre-allocate distance (Source to all Targets)
    distance = pd.Series(index=target.index)

    # Shape auxiliary matrix to the target sample space
    aux = target.loc[target.index, g]

    # Reshape numerical values for distance calculation
    for num in numerical:
        aux.loc[:, num] = source.loc[num]*np.ones([len(target), 1])

    # Absolute difference
    aux.loc[:, numerical] = abs(aux.loc[:, numerical] - target.loc[:, numerical])

    # Continuous variable missing values are replaced by a +3 difference
    aux.loc[:, num_l].fillna(3, inplace=True)

    # Ordinal Replacement
    ord_aux = pd.isnull(aux.loc[:, ord_l])

    if (ord_aux == True).any().any():
        for ordinal in ord_l:

            # The Source is NaN
            if pd.isnull(source.loc[ordinal]):
                aux.loc[:, ordinal] = np.maximum(np.ones(len(target)), np.ones(len(target)) - target.loc[:, ordinal])

                # Both are NaN
                if pd.isnull(aux.loc[:, ordinal]).any():
                    idx = aux[pd.isnull(aux.loc[:, ordinal]) == True].index.tolist()
                    aux.loc[idx, ordinal] = np.ones([len(idx), 1])

            # The Target could be a NaN
            else:
                # The Target is a NaN
                idx = ord_aux[ord_aux.loc[:, ordinal] == True].index.tolist()
                aux.loc[idx, ordinal] = np.maximum(np.ones(len(idx)), np.ones(len(idx)) - np.ones(len(idx))*source.loc[ordinal])

    for nominal in nom_l:
        # If the Target is not equal to the Source the distance increases by 1
        aux.loc[:, nominal] = aux.loc[:, nominal].ne(target.loc[:, nominal])

    # Store the distance from Source to al Targets
    distance.loc[target.index] = np.sum(aux.values, axis=1)

    # Average distance (over attributes)
    distance = distance/float(len(aux.columns))

    if s_idx in target.index:
        # Distance from Source to Source should not be considered
        distance.drop(s_idx, inplace=True)

    # Rank the neighbors in ascending order of distance
    distance.sort()

    return distance


def cast_vote(risk, label, protected, unprotected):
        vote_p = label.loc[protected.index[0:32]] == risk
        vote_u = label.loc[unprotected.index[0:32]] == risk

        votes = pd.DataFrame(index=['prot', 'unprot'], columns=[8, 16, 32])
        for k in [8, 16, 32]:
            votes.loc['prot', k] = np.divide(np.sum(vote_p.values[0:k]), float(k))
            votes.loc['unprot', k] = np.divide(np.sum(vote_u.values[0:k]), float(k))

        vote_diff = (votes.loc['prot', :] - votes.loc['unprot', :]).values

        return vote_diff


def d_calc(target_v, source_v, i_var, o_var):
    d_tuple = 0
    # Calculate the distance for each variable type
    for feat in range(len(target_v)):

        # Interval Variable
        if feat in i_var:
            # NaN or missing value
            if target_v[feat] is np.nan or target_v[feat] == '?' or source_v[feat] is np.nan or source_v[feat] == '?':
                d_tuple += 3

            # The absolute difference
            else:
                d = np.subtract(target_v[feat], source_v[feat])
                d_tuple += np.absolute(d)

        # Ordinal Variable
        elif feat in o_var:
            # Both are missing
            if (target_v[feat] is np.nan or target_v[feat] == '?') and (source_v[feat] is np.nan or source_v[feat] == '?'):
                d_tuple += 1

            # Only one is missing
            elif (target_v[feat] is np.nan or target_v[feat] == '?') and not (source_v[feat] is np.nan or source_v[feat] == '?'):
                d_tuple += np.max(source_v[feat], np.subtract(1, source_v[feat]))
            elif (source_v[feat] is np.nan or source_v[feat] == '?') and not (target_v[feat] is np.nan or target_v[feat] == '?'):
                d_tuple += np.max(target_v[feat], np.subtract(1, target_v[feat]))

            # The absolute difference
            else:
                d = np.subtract(target_v[feat], source_v[feat])
                d_tuple += np.absolute(d)

        # Nominal Variable
        else:
            # NaN or missing value
            if target_v[feat] is np.nan or target_v[feat] == '?' or source_v[feat] is np.nan or source_v[feat] == '?':
                d_tuple += 1

            # The tuple does not share the same attribute value
            elif not target_v[feat] == source_v[feat]:
                d_tuple += 1

    d_tuple = np.divide(d_tuple, len(source_v))

    return d_tuple


def data_plot(data, bins, fig, limits, line_spec, label, save_plot, filename):

    plt.figure(fig)
    for column, i in zip(data.columns.tolist(), range(len(line_spec[0]))):

        # Sort the data in descending order
        t = data.sort(column, axis=0, ascending=False)[column]

        # Create bins for storing the data
        t_un = np.linspace(min(t), max(t), bins)

        # Place the cumulative fraction of tuples in the bins
        y = []
        for bin in t_un:

            y_int = 0
            for sample in t:

                if sample >= bin:
                    y_int += 1

            y.append(y_int)

        # Normalize the Fraction of Tuples
        y = np.divide(y, float(len(t)))

        # Plot the data
        plt.plot(t_un, y, line_spec[0][i], linewidth=line_spec[1][i])

    plt.axis(limits)
    plt.xlabel('t')
    plt.ylabel('Frac. of tuples (diff $\geq$ t)')
    plt.title(filename)
    plt.legend(label)
    plt.grid()
    if save_plot:
        plt.savefig('../results/' + filename + '.png')


def disc(p_idx, data, data_label, protected, nom_l, num_l, ord_l, class_set, status, diff_data, threshold):

    # Discriminatory label
    L = pd.Series(index=p_idx)

    # Protected group
    for r in p_idx:

        # Negative decision and difference above t
        if data_label.loc[r] == class_set[status] and diff_data.loc[r] >= threshold:
            L[r] = True
        else:
            L[r] = False


    # Identify non-protected attributes
    g = []
    for i in range(len(protected)):
        if protected[i] is not None:
            if i in num_l:
                num_l.remove(i)
            elif i in ord_l:
                ord_l.remove(i)
            elif i in nom_l:
                nom_l.remove(i)
        else:
            g.append(i)

    # Vectorize Categorical Data
    vf = pd.get_dummies(data.loc[p1, nom_l])

    # Concatenate evidence
    d = np.concatenate((vf, data.loc[p1, num_l].values), axis=1)
    d = np.concatenate((d, data.loc[p1, ord_l]), axis=1)
    d = np.array(d).astype(float)

    # Circumvent NaN
    idx = np.where(np.isnan(d))

    if len(idx) >= 1:
        d[idx] = 5

    # Split test and training set (2/3, 1/3)
    d_train, d_test, y_train, y_test = skl.cross_validation.train_test_split(d, L, test_size=0.33)
    y_idx = y_test.index

    # Build Classifier
    p_tree = skt.DecisionTreeClassifier(random_state=0)

    # Fit Data to Tree
    p_tree.fit(d_train, y_train)

    # Predict
    prediction = p_tree.predict(d_test)

    # Calculate Precision and Recall
    abc = np.array([0, 0, 0]).astype(float)

    for x in range(len(y_test.index)):

        # True Positive
        if y_test[y_idx[x]] > 0:

            # Predicted Positive
            if prediction[x] > 0:
                abc[0] += 1

            # False Negative
            else:
                abc[1] += 1

        # True negative
        else:

            # False positive
            if prediction[x] > 0:
                abc[2] += 1


    precision = np.divide(abc[0], abc[0] + abc[2])
    recall = np.divide(abc[0], abc[0] + abc[1])

    print 'Test set:'
    # Precision
    if not np.isnan(precision):
        print 'Precision = ' + str(precision)
    else:
        print 'Precision (Division by zero): No discrimination'

    # Recall
    if not np.isnan(recall):
        print 'Recall = ' + str(recall) + '\n'
    else:
        print 'Recall (Division by zero): No discrimination and no false negatives\n'


    print 'Training + Test set:'
    # Add true positives from the training set (descriptive)
    abc[0] += sum(y_train)
    precision = np.divide(abc[0], abc[0] + abc[2])
    recall = np.divide(abc[0], abc[0] + abc[1])

    # Precision
    if not np.isnan(precision):
        print 'Precision = ' + str(precision)
    else:
        print 'Precision (Division by zero): No discrimination'

    # Recall
    if not np.isnan(recall):
        print 'Recall = ' + str(recall) + '\n'
    else:
        print 'Recall (Division by zero): No discrimination and no false negatives\n'


def usage_msg():
    return '''
            Please provide at least one flag for the analysis:
                    -cf: German Credit, Female Non-single
                    -cm: German Credit, Male, Married, 30 < Age < 60
                    -anw: Adult Census, Non-white person
            save_diff - Save plots and diff files (Boolean, 1 - saves to results folder, 0 - only shows plots, mandatory)'''


parser = argparse.ArgumentParser(description='Discrimination discovery in the German credit score or the Adult census data set', usage=usage_msg())
parser.add_argument(dest='save_diff', action='store', type=int,
                    help='Boolean - Save plots and diff files')
parser.add_argument('-cf', action='store_true', dest='run_f', default=False,
                    help='Run German credit analysis with non-single female as protected attribute')
parser.add_argument('-cm', action='store_true', dest='run_m', default=False,
                    help='Run German credit analysis with married man (30 < age < 60) as protected attribute')
parser.add_argument('-anw', action='store_true', dest='run_a', default=False,
                    help='Run Adult census analysis with non-white person as protected attribute')
args = parser.parse_args()

if args.save_diff == 0:
    args.save_diff = False
else:
    args.save_diff = True

# --------------------------The German Credit Score Data-------------------------
if args.run_f or args.run_m:
    # Interval Variables
    int_var = [1, 4, 7, 12, 15, 17]

    # Ordinal Variables
    ord_var = [0, 5, 6]

    # Nominal Variables
    nom_var = [2, 3, 8, 9, 10, 11, 13, 14, 16, 18, 19]

    # Outcomes: 1 - Low Risk, 2 - High Risk
    outcome = [1, 2]

    # Graph axis
    axis = [-1, 1, -0.05, 1.05]

    gcre, credit_label, mu, sigma = parse_data('../data/german_credit/german.data',
                                               np.arange(start=0, stop=21, step=1),
                                               1,
                                               int_var,
                                               ord_var,
                                               20)

# ------Female non-single
if args.run_f:
    p_female = [None]*8 + ['A92'] + [None]*11

    p1, p2 = discriminate(p_female, gcre)

    t = pd.DataFrame(index=pd.MultiIndex.from_product([p1, outcome]), columns=['8', '16', '32'])

    for sample in p1:

        # Neighbors of the sample in the Protected group
        d_p = rank(sample, gcre.loc[sample, :], gcre.loc[p1, :], p_female, int_var, ord_var, nom_var)

        # Neighbors of the sample in the Unprotected group
        d_u = rank(sample, gcre.loc[sample, :], gcre.loc[p2, :], p_female, int_var, ord_var, nom_var)

        # Compute the Votes
        for risk in outcome:
            t.loc[(sample, risk), :] = cast_vote(risk, credit_label, d_p, d_u)

    # Plot Cumulative t distribution
    graph_label = ['k=8', 'k=16', 'k=32']
    graph_spec = [['k-', 'k--', 'k:'], [2, 1, 1]]

    data_plot(t.loc[(p1, outcome[0]), :], 10, 1, axis, graph_spec, graph_label, args.save_diff, 'credit_female_low')
    data_plot(t.loc[(p1, outcome[1]), :], 10, 2, axis, graph_spec, graph_label, args.save_diff, 'credit_female_high')

    # Discovery Results
    print 'Female non-single'
    disc(p1, gcre, credit_label, p_female, nom_var, int_var, ord_var, outcome, 1, t.loc[(p1, outcome[1]), '32'].reset_index(level=1)['32'], 0.1)

    # Save the data or just display the plots
    if args.save_diff:
        t.loc[(p1, outcome[0]), :].to_csv('../results/credit_female_low.txt')
        t.loc[(p1, outcome[1]), :].to_csv('../results/credit_female_high.txt')

# ------Married Man, 30 < Age < 60
if args.run_m:
    p_male = [None]*8 + ['A94'] + [None]*3 + [[30, 60]] + [None]*7

    p1, p2 = discriminate(p_male, gcre)

    t = pd.DataFrame(index=pd.MultiIndex.from_product([p1, outcome]), columns=['8', '16', '32'])

    for sample in p1:

        # Neighbors of the sample in the Protected group
        d_p = rank(sample, gcre.loc[sample, :], gcre.loc[p1, :], p_male, int_var, ord_var, nom_var)

        # Neighbors of the sample in the Unprotected group
        d_u = rank(sample, gcre.loc[sample, :], gcre.loc[p2, :], p_male, int_var, ord_var, nom_var)

        # Compute the Votes
        for risk in outcome:
            t.loc[(sample, risk), :] = cast_vote(risk, credit_label, d_p, d_u)

    # Plot Cumulative t distribution
    graph_label = ['Low Risk', 'High Risk']
    graph_spec = [['k--', 'k-'], [2, 2]]

    m_data = pd.DataFrame(index=p1)
    m_data['low'] = t.loc[(p1, outcome[0]), '16'].reset_index(level=1)['16']
    m_data['high'] = t.loc[(p1, outcome[1]), '16'].reset_index(level=1)['16']

    data_plot(m_data, 7, 3, axis, graph_spec, graph_label, args.save_diff, 'credit_male')

    # Discovery Results
    print 'Male, married, 30 < Age < 60'
    disc(p1, gcre, credit_label, p_male, nom_var, int_var, ord_var, outcome, 1, t.loc[(p1, outcome[1]), '32'].reset_index(level=1)['32'], 0.1)

    # Save the data or just display the plots
    if args.save_diff:
        t.loc[(p1, outcome[0]), :].to_csv('../results/credit_male_low.txt')
        t.loc[(p1, outcome[1]), :].to_csv('../results/credit_male_high.txt')
# -------------------------------------------------------------------------------

# ------------------------------The Adult Census---------------------------------
if args.run_a:
    # Interval Variables
    int_var = [0, 2, 4, 10, 11, 12]

    # Nominal Variables
    nom_var = [1, 3, 5, 6, 7, 8, 9, 13]

    # Ordinal Variables
    ord_var = []

    # Outcomes: Income > 50k USD or Income <= 50k USD
    decision_set = ['>50K', '<=50K']

    # Graph Settings
    graph_label = ['k=8', 'k=16', 'k=32']
    graph_spec = [['k-', 'k--', 'k:'], [2, 2, 2]]
    axis = [0, 1, -0.05, 0.45]

    adult, adult_label, mu, sigma = parse_data('../data/adult/adult.data',
                                               np.arange(start=0, stop=15, step=1),
                                               0,
                                               int_var,
                                               ord_var,
                                               14)

# ------Married Man, Non-White
    p_adult = [None]*8 + [['Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']] + [None]*5

    p1, p2 = discriminate(p_adult, adult)

    t = pd.DataFrame(index=pd.MultiIndex.from_product([p1, decision_set]), columns=['8', '16', '32'])

    for sample in p1:

        # Neighbors of the sample in the Protected group
        d_p = rank(sample, adult.loc[sample, :], adult.loc[p1, :], p_adult, int_var, ord_var, nom_var)

        # Neighbors of the sample in the Unprotected group
        d_u = rank(sample, adult.loc[sample, :], adult.loc[p2, :], p_adult, int_var, ord_var, nom_var)

        # Compute the Votes
        for risk in decision_set:
            t.loc[(sample, risk), :] = cast_vote(risk, adult_label, d_p, d_u)

    # Set multilevel arrangement
    t.sortlevel(inplace=True, sort_remaining=True)

    # Plot Cumulative t distribution
    data_plot(t.loc[(p1, decision_set[1]), :], 12, 4, axis, graph_spec, graph_label, args.save_diff, 'adult_nonwhite_less_50k')

    # Discovery Results
    print 'Adult non-white with income < 50k'
    disc(p1, adult, adult_label, p_adult, nom_var, int_var, ord_var, decision_set, 1, t.loc[(p1, decision_set[1]), '32'].reset_index(level=1)['32'], 0.1)

    if args.save_diff:
        t.loc[(p1, decision_set[0]), :].to_csv('../results/adult_nonwhite_more_50k.txt')
        t.loc[(p1, decision_set[1]), :].to_csv('../results/adult_nonwhite_less_50k.txt')
# -------------------------------------------------------------------------------

if not args.save_diff:
    plt.show()