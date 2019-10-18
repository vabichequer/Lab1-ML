from sklearn.model_selection import train_test_split

def train_val_test_split(X, y, val_size=0.0, test_size=0.8, random_state=None):
    
    if isinstance(val_size, int):
        nval = val_size
    else:
        nval = int(len(X)*val_size)
    if isinstance(test_size, int):
        ntest = test_size
    else:
        ntest = int(len(X)*test_size)
    
    X_trval, X_test, y_trval, y_test = train_test_split(X, y,
                                                        test_size=ntest,
                                                        random_state=random_state)
    if random_state == None:
        aux = None
    else:
        aux = 54773*random_state
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval,
                                                      test_size=nval,
                                                      random_state=aux)

    return X_train, X_val, X_test, y_train, y_val, y_test


# based and extended from:
# http://scikit-learn-general.narkive.com/qDWR2kGK/how-to-extract-the-decision-tree-rule-of-each-leaf-node-into-pandas-dataframe-query


def get_rules_from_tree(clf, feature_names, class_values, X, y):
    from sklearn.tree._tree import TREE_LEAF
    
    rules = {}
    
    def recurse(node_id, aux_rules=[]):
        left = clf.tree_.children_left[node_id]
        right = clf.tree_.children_right[node_id]

        #Check if this is a decision node
        if left != TREE_LEAF:
            feature = feature_names[clf.tree_.feature[node_id]]
            new_rule = "(" + feature + " {0} " + "%.4f" % clf.tree_.threshold[node_id] + ")"
            recurse(left, aux_rules + [new_rule.format('<=')])
            recurse(right, aux_rules + [new_rule.format('>')])
            return
        else: # Leaf
            rules[node_id] = " and ".join(aux_rules)
            return

    recurse(node_id=0)
    classification_leafs = clf.apply(X)
    extended_rules = {}
    for k,v in rules.items():
        aux1 = y[classification_leafs == k].tolist()
        aux2 = []
        for c, name in zip(clf.classes_, class_values):
            aux3 = aux1.count(c)
            aux2.append((name, aux3, aux3/len(aux1)))
        aux_dict = {}
        extended_rules[k] = [v, aux2]
    
    return extended_rules
