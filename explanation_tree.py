import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
import copy
import scipy.cluster.hierarchy as shc
import re
import ipywidgets as widgets
import cv2
import plotly.graph_objects as go
import umap
import igraph
from igraph import Graph, EdgeSeq
import matplotlib.pyplot as plt
import scipy
from skrules import SkopeRules
from tqdm.auto import tqdm

def format_rule(string, precision=2):
    try:
        return str(round(float(string),precision))
    except:
        return string

def reduce_rule(rule):
    list_rule = rule.split(" and ")
    dict_rules = {}
    for i,rule in enumerate(list_rule):
        rulesplit = rule.split(" ")
        if len(rulesplit) == 3:  #Save the rule
            dict_rules[i] = {'col' : rulesplit[0],
                            'sign' : rulesplit[1],
                            'value' : rulesplit[2]}
        elif len(rulesplit) == 5:  #Re-split the rule (ex : 0 < col1 <= 1) in two (col > 0 and col <= 1)
            dict_rules[i] = {'col' : rulesplit[2],
                            'sign' : '>',
                            'value' : rulesplit[0]}
            
            dict_rules[i+0.5] = {'col' : rulesplit[2],
                                'sign' : '<=',
                                'value' : rulesplit[4]}
    
    dict_new_rules = dict_rules.copy()
    ## First, reduce all the redundant rules (ex : 'col1 <= val1 and col1 <= val2')
    change=True
    while change:
        dict_rules = dict_new_rules.copy()
        change=False
        for rule1 in dict_rules:
            for rule2 in dict_rules:
                if (rule1 < rule2) and (dict_rules[rule1]['col']==dict_rules[rule2]['col']): # rules on the same col
                    if dict_rules[rule1]['sign'] == dict_rules[rule2]['sign']:
                        dict_new_rules[rule1] = {'col':dict_rules[rule1]['col'],
                                            'sign':dict_rules[rule1]['sign'],
                                            'value':str(min(float(dict_rules[rule1]['value']),float(dict_rules[rule2]['value']))) if dict_rules[rule1]['sign']=="<=" else str(max(float(dict_rules[rule1]['value']),float(dict_rules[rule2]['value'])))}
                        del dict_new_rules[rule2]
                        change=True
    
    ## Second, merge rules on same col (ex : 'col1 > val1 and col1 <= val2')
    dict_new_new_rules = dict_new_rules.copy()
    for rule1 in dict_new_rules:
        for rule2 in dict_new_rules:
            if (rule1 < rule2) and (dict_new_rules[rule1]['col']==dict_new_rules[rule2]['col']) and (dict_new_rules[rule1]['sign'] != dict_new_rules[rule2]['sign']): # rules on the same col
                if dict_new_rules[rule1]['sign'] == '==':
                    del dict_new_new_rules[rule2]
                elif dict_new_rules[rule2]['sign'] == '==':
                    try:
                        del dict_new_new_rules[rule1]
                    except:
                        pass                    
                elif (dict_new_rules[rule1]['sign'] == '!=') or (dict_new_rules[rule2]['sign'] == '!='):
                    pass
                else:
                    dict_new_new_rules[rule1] = {'col':dict_new_rules[rule1]['col'],
                                                'sign':'both',
                                                'value':str(dict_new_rules[rule1]['value']) + ' and ' + str(dict_new_rules[rule2]['value'])}
                    del dict_new_new_rules[rule2]

    complete_rule=''
    for rule in dict_new_new_rules:
        if dict_new_new_rules[rule]['sign'] == 'both':
            complete_rule += f'{np.min([float(l) for l in dict_new_new_rules[rule]["value"].split(" and ")])} < {dict_new_new_rules[rule]["col"]} <= {np.max([float(l) for l in dict_new_new_rules[rule]["value"].split(" and ")])} and '
        else:
            complete_rule += f'{dict_new_new_rules[rule]["col"]} {dict_new_new_rules[rule]["sign"]} {np.max([float(l) for l in dict_new_new_rules[rule]["value"].split(" and ")])} and '
    
    return complete_rule[:-5] #delete the last "and"

def make_annotations(pos, text, M, font_size=10, font_color='rgb(0,0,0)'):
        L=len(pos)
        if len(text)!=L:
            raise ValueError('The lists pos and text must have the same len')
        annotations = []
        for k in range(L):
            annotations.append(
                dict(
                    text=text[k], # or replace labels with a different list for the text within the circle
                    x=pos[k][0], y=2*M-pos[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return annotations

def fmeasure(labels, clustering):
    """
    Computes the fmeasure between two partitions
    
    Parameters
    ----------
    labels : 1d array-like
        Array representing the first partition
    
    clustering : 1d array-like
        Array of same size as labels representing the other clustering
        
    Returns
    -------
    fmeasure : float
        The value of the fmeasure between the two partitions.
        0 means they are independant. 1 means they are the same.
    """
    
    fmeasure=0
    c,v = np.unique(labels,return_counts=True)
    for ci, vi in zip(c,v):
        maxf=0
        for ki in np.unique(clustering):
            f=sklearn.metrics.f1_score(np.array(labels)==ci, np.array(clustering)==ki, zero_division=0)
            if f > maxf:
                maxf = f
        fmeasure += (vi/len(labels))*maxf
    return fmeasure

class ExplanationTree:
    def __init__(self, X, explanation, node_list=None, depth=None):
        self.X=X
        self.explanation=explanation
        self.node_list=node_list if node_list is not None else []
        self.depth=None

    def get_children(self, node_number):
        """
        Return the children of a given node in a rules DT
        """
        
        children = [[node_number]]
        parent_node = [node for node in self.node_list if node.number==node_number][0]
        if parent_node.left_child is not None:
            children.append(self.get_children(parent_node.left_child.number))
        if parent_node.right_child is not None:
            children.append(self.get_children(parent_node.right_child.number))
        return [child for childlist in children for child in childlist] # flatten

    def get_n_clusters(self, n_clusters, verbose=1):
        df_rules = self.to_df()
        if n_clusters>df_rules["Rule"].isna().sum():
            if verbose>0:
                print("Enter a value of n_clusters lesser or equal than the maximum numbers of approximated clusters")
                print("Computing maximum number of clusters : ", df_rules["Rule"].isna().sum())
            clusters=df_rules[df_rules["Rule"].isna()].index.to_list()
        elif n_clusters==-1: #return max number of clusters
            clusters=df_rules[df_rules["Rule"].isna()].index.to_list()
        else:
            clusters = [0]
            i=0
            while len(clusters)<n_clusters:
                clusters.append(i+1)
                clusters.append(i+2)
                clusters.remove(df_rules.loc[i+2,"Parent cluster"])
                i+=2

        return clusters

    def get_labels_from_clusters(self, clusters):
        labels = pd.Series(0, index=self.X.index, dtype="int")
        for c in clusters:
            labels+=self.X.eval([node.complete_rule for node in self.node_list if node.number==c][0])*c
        return labels

    def get_n_labels(self, n_clusters, verbose=1):
        return self.get_labels_from_clusters(self.get_n_clusters(n_clusters, verbose=verbose))

    def approximate_partition_skrules(self, n_clusters, max_nb_rules=3):
        warnings.filterwarnings("ignore", category=FutureWarning)
        labels_skrules=pd.Series(0, index=self.X.index, dtype="int")
        # df_explanation = pd.DataFrame(self.explanation.values, index=self.X.index, columns=self.X.columns)
        for i in range(1,n_clusters+1):
            skrules = SkopeRules(max_depth=max_nb_rules, precision_min=0, recall_min=0, feature_names=self.X.columns).fit(self.X, (shc.fcluster(self.link, n_clusters, criterion="maxclust")==i)*1)
            labels_skrules.loc[self.X.eval(skrules.rules_[0][0])]=i
        warnings.filterwarnings("default", category=FutureWarning)
        return labels_skrules


    def compare_f_measure(self, link, draw=True):
        df_rules = self.to_df()
        df_fmeasure = pd.DataFrame(columns=["n_clusters","fmeasure_dt","fmeasure_skoperules"])
        for n_clusters in tqdm(range(2, df_rules["Rule"].isna().sum())):
            df_fmeasure = pd.concat([df_fmeasure, pd.DataFrame([[n_clusters,
                                                                fmeasure(self.get_n_clusters(n_clusters=n_clusters),
                                                                         shc.fcluster(link, n_clusters, criterion='maxclust')
                                                                         ),
                                                                fmeasure(self.approximate_partition_skrules(link, n_clusters=n_clusters),
                                                                         shc.fcluster(link, n_clusters, criterion='maxclust')
                                                                         )]],
                                                              columns=df_fmeasure.columns)],
                                    ignore_index=True)

        if draw:
            plt.plot(df_fmeasure["n_clusters"], df_fmeasure["fmeasure_dt"],label="dt")
            plt.plot(df_fmeasure["n_clusters"], df_fmeasure["fmeasure_skoperules"],label="skoperules")
            plt.show()


        return df_fmeasure

class HierarchicalExplanationTree(ExplanationTree):
    def __init__(self, X, explanation, node_list=None, depth=None, link=None):
        super().__init__(X, explanation, node_list=node_list, depth=depth)
        self.link=link

    def _append_dt_node_recur(self, dt, id_node, parent, node_type):
        is_split_node = dt.tree_.children_left[id_node] != dt.tree_.children_right[id_node]
        rule = f'{self.X.columns[dt.tree_.feature[id_node]]} <= {dt.tree_.threshold[id_node]}' if is_split_node else None
        node = HierarchicalNode(number=id_node,
                    parent=parent,
                    rule=rule,
                    node_type=node_type)
        node.set_complete_rule(first_call=True)
        self.node_list.append(node)

        if is_split_node:
            left_node = self._append_dt_node_recur(dt, dt.tree_.children_left[id_node], parent=node, node_type="Left")
            right_node = self._append_dt_node_recur(dt, dt.tree_.children_right[id_node], parent=node, node_type="Right")
            node.left_child = left_node
            node.right_child = right_node

        return node

    def approximate_dt(self, n_clusters):
        dt=DecisionTreeClassifier(max_leaf_nodes=n_clusters).fit(self.X, shc.fcluster(self.link, n_clusters, criterion="maxclust"))
        n_nodes = dt.tree_.node_count
        # print("n_nodes : ",n_nodes)

        self._append_dt_node_recur(dt, 0, None, "Root")

    def approximate_dt_target(self, y, n_clusters):
        dt=DecisionTreeClassifier(max_leaf_nodes=n_clusters).fit(self.X, y)
        n_nodes = dt.tree_.node_count
        # print("n_nodes : ",n_nodes)

        self._append_dt_node_recur(dt, 0, None, "Root")

    def approximate_hc(self, limit):
        nb=0
        root_node = HierarchicalNode(number=nb,
                         complete_rule=f'{self.X.columns[0]} == {self.X.columns[0]}',
                         rule="None", mask=np.full(self.X.shape[0], True),
                         node_type="Root",
                         f1score=1,
                         parent=None,
                         left_child=None,
                         right_child=None)
        self.node_list.append(root_node)

        for i in range(2, limit):
            labels = shc.fcluster(self.link, i-1, criterion="maxclust")
            labels2 = shc.fcluster(self.link, i, criterion="maxclust")

            for label in np.unique(labels):
                if len(np.unique(labels2[labels==label]))==2:
                    cluster_to_split=label
                    mask_to_split=labels==label
                    break
            
            # print("mask to split : ",mask_to_split)
            # print("mask first node", self.node_list[0].mask)
            parent_node = [node for node in self.node_list if np.all(node.mask==mask_to_split)][0]
            unique_labels, labelsbin = np.unique(labels2[self.X.eval(parent_node.complete_rule)], return_inverse=True) #Puts labels in 0 and 1
            if len(unique_labels)>1:
                labelsdt = (labels2[self.X.eval(parent_node.complete_rule)]==max(labels2[parent_node.mask], key=list(labels2[parent_node.mask]).count))
                dt = sklearn.tree.DecisionTreeClassifier(max_leaf_nodes=2)
                dt.fit(self.X.query(parent_node.complete_rule), labelsdt)
                if len(dt.tree_.value)<2:
                    print("oops")
                    print("cluster to split : ", parent_node.number)
                    print("dt.tree.value : ",dt.tree_.value)
                    print("labelsdt : ", labelsdt)
                rule = f"{self.X.columns[dt.tree_.feature[0]]} <= {format_rule(dt.tree_.threshold[0],3)}"
                parent_node.rule = rule
                full_rule = parent_node._compute_complete_rule(first_call=False)
                if len(np.unique(self.X.query(full_rule)[self.X.columns[dt.tree_.feature[0]]]))==1:
                    parent_node.rule = re.sub(" <= .*", " == "+str(np.unique(self.X.query(full_rule)[self.X.columns[dt.tree_.feature[0]]])[0]),rule)

                masks_leafs = [labels2 == labels2[parent_node.mask][labels2[parent_node.mask] != max(labels2[parent_node.mask], key=list(labels2[parent_node.mask]).count)][0],
                               labels2 == max(labels2[parent_node.mask], key=list(labels2[parent_node.mask]).count)]
                nb+=1
                left_node = HierarchicalNode(number=nb,
                                 mask=masks_leafs[np.argmax(dt.tree_.value[1][0])],
                                 parent=parent_node,
                                 node_type="Left"
                                 )
                self.node_list.append(left_node)
                left_node.set_complete_rule()
                left_node.compute_f1score(self.X)
                parent_node.left_child=left_node

                nb+=1
                right_node = HierarchicalNode(number=nb,
                                 mask=masks_leafs[np.argmax(dt.tree_.value[2][0])],
                                 parent=parent_node,
                                 node_type="Right"
                                 )
                self.node_list.append(right_node)
                right_node.set_complete_rule()
                right_node.compute_f1score(self.X)
                parent_node.right_child=right_node

    def approximate_hc_2(self, limit):
        """
        We remove mistake at each step by considering only the points that follow the rules so far
        """

        nb=0
        root_node = HierarchicalNode(number=nb,
                         complete_rule=f'{self.X.columns[0]} == {self.X.columns[0]}',
                         rule="None", mask=np.full(self.X.shape[0], True),
                         node_type="Root",
                         f1score=1,
                         parent=None,
                         left_child=None,
                         right_child=None)
        self.node_list.append(root_node)

        for i in range(2, limit):
            labels = shc.fcluster(self.link, i-1, criterion="maxclust")
            labels2 = shc.fcluster(self.link, i, criterion="maxclust")

            for label in np.unique(labels):
                if len(np.unique(labels2[labels==label]))==2:
                    cluster_to_split=label
                    mask_to_split=labels==label
                    break
            
            sub_clusters, counts = np.unique(labels2[labels==label], return_counts=True)
            # print(f"\n\n\nMask to split {self.X[mask_to_split].index}")
            # print(f"First subcluster : {sub_clusters[0]} => {self.X.index[labels2==sub_clusters[0]]}")
            # print(f"Second subcluster : {sub_clusters[1]} => {self.X.index[labels2==sub_clusters[1]]}")
            # print(f"Sub_clusters counts : {sub_clusters} : {counts}")
            parent_node = [node for node in self.node_list if np.all(node.mask==mask_to_split)][0]
            # print(f"Parent complete rule : ",parent_node.complete_rule)
            unique_labels, labelsbin = np.unique(labels2[self.X.eval(parent_node.complete_rule)], return_inverse=True) #Puts labels in 0 and 1
            if len(unique_labels)>1:
                labelsdt = labels2[self.X.eval(parent_node.complete_rule) & (labels==label)]==sub_clusters[np.argmax(counts)]
                dt = sklearn.tree.DecisionTreeClassifier(max_leaf_nodes=2)
                dt.fit(self.X[self.X.eval(parent_node.complete_rule) & (labels==label)], labelsdt)
                # indexes_missclassified = self.X[self.X.eval(parent_node.complete_rule) & (labels==label)][dt.predict(self.X[self.X.eval(parent_node.complete_rule) & (labels==label)]) != labelsdt].index
                # print("missclasified indexes : ", indexes_missclassified)
                # print("dt value : ",dt.tree_.value)
                if len(dt.tree_.value)<2:
                    print("oops")
                    print(f"First subcluster : {sub_clusters[0]} => {self.X.index[labels2==sub_clusters[0]]}")
                    print(f"Second subcluster : {sub_clusters[1]} => {self.X.index[labels2==sub_clusters[1]]}")
                    print(f"Parent node members : {self.X[self.X.eval(parent_node.complete_rule)].index}")
                    print("cluster to split : ", parent_node.number)
                    print("dt.tree.value : ",dt.tree_.value)
                    print("labelsdt : ", labelsdt)
                    return -2
                parent_node.dt=dt
                rule = f"{self.X.columns[dt.tree_.feature[0]]} <= {format_rule(dt.tree_.threshold[0],3)}"
                parent_node.rule = rule
                full_rule = parent_node._compute_complete_rule(first_call=False)
                if len(np.unique(self.X.query(full_rule)[self.X.columns[dt.tree_.feature[0]]]))==1:
                    parent_node.rule = re.sub(" <= .*", " == "+str(np.unique(self.X.query(full_rule)[self.X.columns[dt.tree_.feature[0]]])[0]),rule)

                masks_leafs = [labels2 == sub_clusters[1-np.argmax(counts)], #2
                               labels2 == sub_clusters[np.argmax(counts)]] #1

                contingency = sklearn.metrics.confusion_matrix(labelsdt, dt.predict(self.X[self.X.eval(parent_node.complete_rule) & (labels==label)]))

                # print("Left node indexes : ", self.X[masks_leafs[np.argmax(dt.tree_.value[1][0])]].index)
                # print("Right node indexes : ", self.X[masks_leafs[np.argmax(dt.tree_.value[2][0])]].index)
                # print("\n")
                nb+=1
                left_node = HierarchicalNode(number=nb,
                                 mask=masks_leafs[np.argmax(dt.tree_.value[1][0])],
                                 parent=parent_node,
                                 node_type="Left"
                                 )
                self.node_list.append(left_node)
                left_node.set_complete_rule()
                left_node.compute_f1score(self.X)
                parent_node.left_child=left_node

                nb+=1
                right_node = HierarchicalNode(number=nb,
                                 mask=masks_leafs[np.argmax(dt.tree_.value[2][0])],
                                 parent=parent_node,
                                 node_type="Right"
                                 )
                self.node_list.append(right_node)
                right_node.set_complete_rule()
                right_node.compute_f1score(self.X)
                parent_node.right_child=right_node
            else:
                return -1

    def approximate_hc_3(self, limit):
        """
        Exhaustive search with unique values
        """
        nb=0
        root_node = HierarchicalNode(number=nb,
                         complete_rule=f'{self.X.columns[0]} == {self.X.columns[0]}',
                         rule="None", mask=np.full(self.X.shape[0], True),
                         node_type="Root",
                         f1score=1,
                         parent=None,
                         left_child=None,
                         right_child=None)
        self.node_list.append(root_node)

        for i in range(2, limit):
            labels = shc.fcluster(self.link, i-1, criterion="maxclust")
            labels2 = shc.fcluster(self.link, i, criterion="maxclust")

            for label in np.unique(labels):
                if len(np.unique(labels2[labels==label]))==2:
                    cluster_to_split=label
                    mask_to_split=labels==label
                    break
            
            sub_clusters, counts = np.unique(labels2[labels==label], return_counts=True)
            parent_node = [node for node in self.node_list if np.all(node.mask==mask_to_split)][0]
            unique_labels, labelsbin = np.unique(labels2[self.X.eval(parent_node.complete_rule)], return_inverse=True) #Puts labels in 0 and 1
            if len(unique_labels)>1:
                labelsdt = labels2[self.X.eval(parent_node.complete_rule) & (labels==label)]==sub_clusters[np.argmax(counts)]
                if len(np.unique(labelsdt))>1:
                    dt = sklearn.tree.DecisionTreeClassifier(max_leaf_nodes=2)
                    dt.fit(self.X[self.X.eval(parent_node.complete_rule) & (labels==label)], labelsdt)
                    best_split = self._find_best_split(self.X[self.X.eval(parent_node.complete_rule) & (labels==label)], labelsdt)
                    dt.tree_.feature[0] = self.X.columns.get_loc(best_split['feature'])
                    dt.tree_.threshold[0] = best_split['value']
                    # Need to add 0 and 1 in case of pure leaves before substracting it
                    dt.tree_.value[1][0] = np.unique(np.concatenate([[0,1],labelsdt[self.X[self.X.eval(parent_node.complete_rule) & (labels==label)].eval(f"{best_split['feature']} <= {best_split['value']}")]]), return_counts=True)[1] - 1
                    dt.tree_.value[2][0] = np.unique(np.concatenate([[0,1],labelsdt[self.X[self.X.eval(parent_node.complete_rule) & (labels==label)].eval(f"{best_split['feature']} > {best_split['value']}")]]), return_counts=True)[1] - 1
                    # if len(dt.tree_.value)<2:
                    #     print("oops")
                    #     print(f"First subcluster : {sub_clusters[0]} => {self.X.index[labels2==sub_clusters[0]]}")
                    #     print(f"Second subcluster : {sub_clusters[1]} => {self.X.index[labels2==sub_clusters[1]]}")
                    #     print(f"Parent node members : {self.X[self.X.eval(parent_node.complete_rule)].index}")
                    #     print("cluster to split : ", parent_node.number)
                    #     print("dt.tree.value : ",dt.tree_.value)
                    #     print("labelsdt : ", labelsdt)
                    #     return -2
                    parent_node.dt=dt
                    rule = f"{self.X.columns[dt.tree_.feature[0]]} <= {format_rule(dt.tree_.threshold[0],3)}"
                    parent_node.rule = rule
                    full_rule = parent_node._compute_complete_rule(first_call=False)
                    if len(np.unique(self.X.query(full_rule)[self.X.columns[dt.tree_.feature[0]]]))==1:
                        parent_node.rule = re.sub(" <= .*", " == "+str(np.unique(self.X.query(full_rule)[self.X.columns[dt.tree_.feature[0]]])[0]),rule)

                    masks_leafs = [labels2 == sub_clusters[1-np.argmax(counts)], #2
                                labels2 == sub_clusters[np.argmax(counts)]] #1

                    contingency = sklearn.metrics.confusion_matrix(labelsdt, dt.predict(self.X[self.X.eval(parent_node.complete_rule) & (labels==label)]))

                    # print("Left node indexes : ", self.X[masks_leafs[np.argmax(dt.tree_.value[1][0])]].index)
                    # print("Right node indexes : ", self.X[masks_leafs[np.argmax(dt.tree_.value[2][0])]].index)
                    # print("\n")
                    nb+=1
                    left_node = HierarchicalNode(number=nb,
                                    mask=masks_leafs[np.argmax(dt.tree_.value[1][0])],
                                    parent=parent_node,
                                    node_type="Left"
                                    )
                    self.node_list.append(left_node)
                    left_node.set_complete_rule()
                    left_node.compute_f1score(self.X)
                    parent_node.left_child=left_node

                    nb+=1
                    right_node = HierarchicalNode(number=nb,
                                    mask=masks_leafs[np.argmax(dt.tree_.value[2][0])],
                                    parent=parent_node,
                                    node_type="Right"
                                    )
                    self.node_list.append(right_node)
                    right_node.set_complete_rule()
                    right_node.compute_f1score(self.X)
                    parent_node.right_child=right_node

                else:
                    return -2
            else:
                return -1
     
    def _find_best_split(self, X, labels):
        best_split = {"feature":None,
            "value":None, 
            "mistakes":np.inf}

        for col in X.columns:
            for val in np.unique(X[col]):
                mistakes = min(np.sum(X.eval(f'{col} <= {val}') == labels), np.sum(X.eval(f'{col} > {val}') == labels))
                if mistakes < best_split["mistakes"]:
                        best_split.update({"feature":col,
                                        "value":val,
                                        "mistakes":mistakes})

        return best_split

    def approximate_hc_4(self, limit):
        """
        New stopping criterion exploring other nodes when one crashes
        """
        self.X.columns = self.X.columns.str.replace(' ','_')
        nb=0
        root_node = HierarchicalNode(number=nb,
                         complete_rule=f'{self.X.columns[0]} == {self.X.columns[0]}',
                         rule="None", mask=[np.full(self.X.shape[0], True)],
                         node_type="Root",
                         f1score=1,
                         parent=None,
                         left_child=None,
                         right_child=None)
        root_node.compute_dispersion(self.explanation)
        self.node_list.append(root_node)

        if limit==-1:
            limit=self.link.shape[0]
        for i in tqdm(range(2, limit), desc="Approximating hierarchical clustering"):
            labels = shc.fcluster(self.link, i-1, criterion="maxclust")
            labels2 = shc.fcluster(self.link, i, criterion="maxclust")

            for label in np.unique(labels):
                if len(np.unique(labels2[labels==label]))==2:
                    cluster_to_split=label
                    mask_to_split=labels==label
                    break
            
            #print(labels2[labels==label])
            # print("mask to split : ", mask_to_split)
            # print("mask to split numbers : ", mask_to_split.sum())
            # print("mask to split index : ", np.where(np.array(mask_to_split)==True)[0])
            sub_clusters, counts = np.unique(labels2[labels==label], return_counts=True)
            parent_node = self._find_parent_node(mask_to_split)
            labelsdt = labels2[self.X.eval(parent_node.complete_rule) & (labels==label)]==sub_clusters[np.argmax(counts)]
            if len(np.unique(labelsdt))<2: #stopping criterion
                parent_node.mask.append(labels2 == sub_clusters[np.argmax(counts)])
                if len(counts)>1:
                    parent_node.mask.append(labels2 == sub_clusters[1-np.argmax(counts)])
            else:
                dt = sklearn.tree.DecisionTreeClassifier(max_leaf_nodes=2)
                dt.fit(self.X[self.X.eval(parent_node.complete_rule) & (labels==label)], labelsdt)
                parent_node.dt=dt
                rule = f"{self.X.columns[dt.tree_.feature[0]]} <= {format_rule(dt.tree_.threshold[0],3)}"
                parent_node.rule = rule
                full_rule = parent_node._compute_complete_rule(first_call=False)
                if len(np.unique(self.X.query(full_rule)[self.X.columns[dt.tree_.feature[0]]]))==1:
                    parent_node.rule = re.sub(" <= .*", " == "+str(np.unique(self.X.query(full_rule)[self.X.columns[dt.tree_.feature[0]]])[0]),rule)

                masks_leafs = [labels2 == sub_clusters[1-np.argmax(counts)], #2
                            labels2 == sub_clusters[np.argmax(counts)]] #1

                # contingency = sklearn.metrics.confusion_matrix(labelsdt, dt.predict(self.X[self.X.eval(parent_node.complete_rule) & (labels==label)]))

                parent_node.compute_fisher_p()

                nb+=1
                left_node = HierarchicalNode(number=nb,
                                mask=[masks_leafs[np.argmax((dt.tree_.value/dt.tree_.value[0][0])[1][0])]],
                                parent=parent_node,
                                node_type="Left"
                                )
                self.node_list.append(left_node)
                left_node.set_complete_rule()
                left_node.compute_f1score(self.X)
                left_node.compute_dispersion(self.explanation)
                parent_node.left_child=left_node

                nb+=1
                right_node = HierarchicalNode(number=nb,
                                mask=[masks_leafs[np.argmax((dt.tree_.value/dt.tree_.value[0][0])[2][0])]],
                                parent=parent_node,
                                node_type="Right"
                                )
                self.node_list.append(right_node)
                right_node.set_complete_rule()
                right_node.compute_f1score(self.X)
                right_node.compute_dispersion(self.explanation)
                parent_node.right_child=right_node

    def _find_parent_node(self, mask_to_split):
        for node in self.node_list:
            for mask in node.mask:
                if np.all(mask==mask_to_split):
                    return node
        raise Exception("Parent node not found")

    def to_df(self):
        df = pd.DataFrame(columns=["Parent cluster",
                                   "Node type",
                                   "Cluster size",
                                   "HC Cluster size",
                                   "Rule",
                                   "Global F1",
                                   "Dispersion",
                                   "Fisher_p",
                                   "Complete rule"])
        
        for node in self.node_list:
            df = pd.concat([df, pd.DataFrame([[node.parent.number if node.parent is not None else -1,
                                               node.node_type,
                                               self.X.eval(node.complete_rule).sum(),
                                               node.mask[0].sum() if node.mask is not None else None,
                                               node.rule,
                                               node.f1score,
                                               node.dispersion,
                                               node.fisher_p,
                                               node.complete_rule]],
                                             columns=df.columns,
                                             index=[node.number])])
        return df.sort_index()

    def compare_f_measure(self, draw=True):
        return super().compare_f_measure(self.link, draw=draw)
        
    # def compare_skope_rules(self):

    def plotly_tree_dt_interactive(self, title="Interactive tree"):

        axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

        reducer = umap.UMAP()
        reducer.fit(self.explanation.values)
        df_rules = self.to_df()
        df_rules["Global F1"] = df_rules["Global F1"].fillna(0)
        df_rules["indexes"] = df_rules.apply(lambda x: self.X[self.X.eval(x["Complete rule"]).values].index.to_list(), axis=1)
        df_rules["confusion matrix"] = [node.dt.tree_.value[1:].reshape((2,2)).astype(int).tolist() if (node.dt is not None and node.dt.tree_.value.shape[0]>2) else None for node in self.node_list]
        g2 = Graph()
        g2.add_vertices(df_rules.shape[0])
        g2.add_edges([(df_rules.index.get_loc(row["Parent cluster"]),df_rules.index.get_loc(index)) for index, row in df_rules.iloc[1:].iterrows()])

        lay = g2.layout("rt",mode="in", root=[0])

        nr_vertices=df_rules.shape[0]
        position = {k: lay[k] for k in range(nr_vertices)}
        v_label = df_rules["Rule"].astype(str).replace("None","")
        Y = [lay[k][1] for k in range(nr_vertices)]
        M = max(Y)

        es = EdgeSeq(g2)
        E = [e.tuple for e in g2.es] # list of edges
        L = len(position)
        Xn = [position[k][0] for k in range(L)]
        Yn = [2*M-position[k][1] for k in range(L)]
        Xe = []
        Ye = []
        for edge in E:
            Xe+=[position[edge[0]][0],position[edge[1]][0], None]
            Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

        labels = v_label.reset_index(drop=True).str.split(" ").str.join("<br>")

        print(labels)

        # df_rules = df_rules.loc[:,["Parent cluster",    #0
        #                            "Node type",         #1
        #                            "Cluster size",      #2
        #                            "HC Cluster size",   #3
        #                            "Rule",              #4
        #                            "Global F1",         #5
        #                            "Dispersion",        #6
        #                            "fisher_p"           #7
        #                            "Complete rule"]]    #8

        hovercolumns = ["Parent cluster",
                        "Cluster size",
                        "confusion matrix",
                        "Global F1",
                        "Dispersion",
                        "Fisher_p",
                        "Complete rule"]

        fig = go.Figure(layout={"height":M*200,"width":1500}) #(np.nanmax(np.array(Xe,dtype=np.float64))-np.nanmin(np.array(Xe,dtype=np.float64)))*150})
        fig.add_trace(go.Scatter(x=Xe,
                        y=Ye,
                        mode='lines',
                        line=dict(color='rgb(210,210,210)', width=1),
                        hoverinfo='none'
                        ))
        fig.add_trace(go.Scatter(x=Xn,
                                y=Yn,
                                mode='markers',
                                name='nodes',
                                marker=dict(symbol='square',
                                            size=df_rules["Cluster size"].astype(int).values,
                                            sizemode='area',
                                            sizeref=0.1,
                                            # size=50,
                                            color=df_rules["Dispersion"].astype(float).values,    #'#DB4551',
                                            colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(0,0,255)']],
                                            colorbar={"title": 'Dispersion'},
                                            showscale=True,
                                            line=dict(color='rgba(0,0,0,1)', width=1)
                                        ),
                                text=labels,
                                customdata=df_rules.values,
                                hovertemplate='<br>'.join([col+': %{customdata['+str(df_rules.columns.get_loc(col))+']}' for col in hovercolumns]),
                                #hovertemplate='Cluster size: %{customdata['+str(df_rules.columns.get_loc("Cluster size"))+']}<br>Global F1: %{customdata[5]:.3f}<br>Dispersion: %{customdata[6]:.3f}<br>Complete rule: %{customdata[7]}<br>Contingency matrix: %{customdata[9]}<br>Fisher test p-value: %{customdata[10]}',
                                opacity=0.8
                        ))

        fig.update_layout(title=title,
                        annotations=make_annotations(position, labels, M),
                        font_size=12,
                        showlegend=False,
                        xaxis=axis,
                        yaxis=axis,
                        margin=dict(l=40, r=40, b=85, t=100),
                        hovermode='closest',
                        plot_bgcolor='rgb(248,248,248)'
                        )
        
        #out = widgets.Output()
        f = go.FigureWidget(fig)
        decision = widgets.Image(format="png", layout={'border':'1px solid'})
        out = widgets.Output()
        def show_members(trace, points, selector, df_rules=df_rules):
            # out.clear_output()
            # with out:
            #     print(trace['customdata'][points.point_inds[0]])
            #     print("Rule : ")
            #     print(self.X.eval(trace['customdata'][points.point_inds[0]][hovercolumns.index("Complete rule")]))
            #     print(self.X.eval(trace['customdata'][points.point_inds[0]][index_complete_rule]))
            index_complete_rule=df_rules.columns.get_loc("Complete rule")
            plt.figure()
            mask = self.X.eval(trace['customdata'][points.point_inds[0]][index_complete_rule])
            #mask.index = [self.X.index.get_loc(i) for i in self.X[mask].index]
            if self.explanation.values.shape[1]>2:
                plt.scatter(reducer.embedding_[:,0], reducer.embedding_[:,1])
                plt.scatter(reducer.embedding_[mask,0],
                            reducer.embedding_[mask,1])
            else:
                plt.scatter(self.X.iloc[:,0], self.X.iloc[:,1])
                plt.scatter(self.X.loc[mask].iloc[:,0],
                            self.X.loc[mask].iloc[:,1])
            plt.title(trace['customdata'][points.point_inds[0]][index_complete_rule])
            plt.savefig("decision_members.png",bbox_inches="tight")
            file = open("decision_members.png","rb")
            image = file.read()
            im = cv2.imread("decision_members.png")
            decision.value=image
            decision.height=im.shape[0]
            decision.width=im.shape[1]
            plt.close()
            
            with out:
                out.clear_output()
            f.update_traces(selectedpoints=[df_rules.index.get_loc(i) for i in self.get_children(df_rules.iloc[points.point_inds[0]].name)],
                            mode='markers', selector={"mode":"markers"}, unselected={'marker': { 'color':'grey','opacity': 0.1}}
                            )
            
        f.data[1].on_click(show_members)
    
        return widgets.VBox([f,out,decision])
    
    def prune(self, threshold=0.05, verbose=0):
        """
        First call of the pruning algorithm

        threshold is the statistical threshold for Fisher test significance before Bonferroni correction
        """

        n_nodes = len([node for node in self.node_list if node.left_child is not None])
        self._recur_prune(self.node_list[0], threshold=threshold/n_nodes, verbose=verbose)  # Bonferroni correction for Fisher test

    def _recur_prune(self, node, threshold, verbose=0):
        if node.left_child is None: #leaf
            return True
        else:
            left_prune = self._recur_prune(node.left_child, threshold=threshold)
            right_prune = self._recur_prune(node.right_child, threshold=threshold)
            if left_prune and right_prune:
                if (node.fisher_p[1] > threshold): #children have been pruned and fisher test is non-significant
                    if verbose > 0: print("Pruning node ", node.number)
                    #Prune node
                    self._prune_node(node)
                    return True
                else:
                    return False #Do not prune, and do not prune any parent
            else:
                return False #Do not prune, and do not prune any parent
            
    def _prune_node(self, node):
        self.node_list = [elem for elem in self.node_list if elem.number not in {node.left_child.number, node.right_child.number}]
        #Update node
        node.left_child = None
        node.right_child = None
        node.fisher_p = None
        node.rule = None


class UnsupervisedExplanationTree(ExplanationTree):
    def ud3(self, min_cover, rule_precision=2, parent_node=None):
        if parent_node is None: #Create root
            parent_node = UnsupervisedNode(number=0,
                                complete_rule=f'{self.X.columns[0]} == {self.X.columns[0]}',
                                node_type="Root")
            self.node_list.append(parent_node)

        best_split = {"feature":None,
                    "value":None, 
                    "criterion":0}
        
        
        X = self.X.query(parent_node.complete_rule)
        df_explanation = pd.DataFrame(self.explanation.values, columns=self.X.columns, index=self.X.index).loc[X.index]

        for col in X.columns:
            for val in np.unique(X[col]):
                if (X.query(f'{col} <= {val}').shape[0] >= min_cover) and (X.query(f'{col} > {val}').shape[0] >= min_cover):
                    Q = np.linalg.norm(df_explanation[X[col]<=val].mean() - df_explanation[X[col]>val].mean()) # euclidean distance between the means of the two children nodes
                    if Q>best_split["criterion"]:
                        best_split.update({"feature":col,
                                        "value":val,
                                        "criterion":Q})
        
        if best_split["value"] is not None:
            rule = f'{best_split["feature"]} <= {best_split["value"]}'#format_full_rule(f'{best_split["feature"]} <= {best_split["value"]}', precision=rule_precision)
            parent_node.rule = rule
            parent_node.criterion = best_split["criterion"]
            left_node = UnsupervisedNode(number=len(self.node_list),
                            parent=parent_node,
                            node_type="Left"
                            )
            self.node_list.append(left_node)
            left_node.set_complete_rule()
            parent_node.left_child=left_node

            self.ud3(min_cover=min_cover, rule_precision=rule_precision, parent_node=left_node)

            right_node = UnsupervisedNode(number=len(self.node_list),
                            parent=parent_node,
                            node_type="Right"
                            )
            self.node_list.append(right_node)
            right_node.set_complete_rule()
            parent_node.right_child=right_node

            self.ud3(min_cover=min_cover, rule_precision=rule_precision, parent_node=right_node)

    def to_df(self):
        df = pd.DataFrame(columns=["Parent cluster",
                                   "Node type",
                                   "Cluster size",
                                   "Rule",
                                   "Criterion",
                                   "Complete rule"])
        
        for node in self.node_list:
            df = pd.concat([df, pd.DataFrame([[node.parent.number if node.parent is not None else -1,
                                               node.node_type,
                                               self.X.eval(node.complete_rule).sum(),
                                               node.rule,
                                               node.criterion,
                                               node.complete_rule]],
                                             columns=df.columns,
                                             index=[node.number])])
                                
        return df

    def plotly_tree_interactive(self, title="Interactive UDtree"):

        axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

        reducer = umap.UMAP()
        reducer.fit(self.explanation.values)
        df_rules = self.to_df()
        g2 = Graph()
        g2.add_vertices(df_rules.shape[0])
        g2.add_edges([(row["Parent cluster"],index) for index, row in df_rules.iloc[1:].iterrows()])

        lay = g2.layout("rt",mode="in", root=[0])

        nr_vertices=df_rules.shape[0]
        position = {k: lay[k] for k in range(nr_vertices)}
        v_label = df_rules["Rule"].astype(str).replace("None","Leaf")
        Y = [lay[k][1] for k in range(nr_vertices)]
        M = max(Y)

        es = EdgeSeq(g2)
        E = [e.tuple for e in g2.es] # list of edges
        L = len(position)
        Xn = [position[k][0] for k in range(L)]
        Yn = [2*M-position[k][1] for k in range(L)]
        Xe = []
        Ye = []
        for edge in E:
            Xe+=[position[edge[0]][0],position[edge[1]][0], None]
            Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

        labels = v_label.str.split(" ").str.join("<br>")

        print(labels)

        fig = go.Figure(layout={"height":M*100,"width":1500}) #(np.nanmax(np.array(Xe,dtype=np.float64))-np.nanmin(np.array(Xe,dtype=np.float64)))*150})
        fig.add_trace(go.Scatter(x=Xe,
                        y=Ye,
                        mode='lines',
                        line=dict(color='rgb(210,210,210)', width=1),
                        hoverinfo='none'
                        ))
        fig.add_trace(go.Scatter(x=Xn,
                                y=Yn,
                                mode='markers',
                                name='nodes',
                                marker=dict(symbol='square-dot',
                                            size=40,
                                            color='#6175c1',    #'#DB4551',
                                            line=dict(color='rgb(50,50,50)', width=1)
                                        ),
                                text=labels,
                                customdata=df_rules.values,
                                hovertemplate='Cluster size: %{customdata[2]}<br>Criterion: %{customdata[4]:.3f}<br>Complete rule: %{customdata[5]}',
                                opacity=0.8
                        ))

        fig.update_layout(title=title,
                        annotations=make_annotations(position, labels, M),
                        font_size=12,
                        showlegend=False,
                        xaxis=axis,
                        yaxis=axis,
                        margin=dict(l=40, r=40, b=85, t=100),
                        hovermode='closest',
                        plot_bgcolor='rgb(248,248,248)'
                        )
        
        #out = widgets.Output()
        f = go.FigureWidget(fig)
        decision = widgets.Image(format="png", layout={'border':'1px solid'})
        out = widgets.Output()
        def show_members(trace, points, selector, df_rules=df_rules):
            plt.figure()
            plt.scatter(reducer.embedding_[:,0], reducer.embedding_[:,1])
            plt.scatter(reducer.embedding_[self.X.eval(trace['customdata'][points.point_inds[0]][5]),0],
                        reducer.embedding_[self.X.eval(trace['customdata'][points.point_inds[0]][5]),1])
            plt.title(trace['customdata'][points.point_inds[0]][5])
            plt.savefig("decision_members.png",bbox_inches="tight")
            file = open("decision_members.png","rb")
            image = file.read()
            im = cv2.imread("decision_members.png")
            decision.value=image
            decision.height=im.shape[0]
            decision.width=im.shape[1]
            
            with out:
                out.clear_output()
                print(points.point_inds[0])
                print(self.get_children(points.point_inds[0]))
            f.update_traces(selectedpoints=self.get_children(points.point_inds[0]),
                            mode='markers', selector={"mode":"markers"}, unselected={'marker': { 'color':'grey','opacity': 0.1}}
                            )
            
        f.data[1].on_click(show_members)
    
        return widgets.VBox([f,out,decision])

class Node:
    def __init__(self, number, complete_rule=None, rule=None, node_type=None, parent=None, left_child=None, right_child=None):
        self.number=number
        self.complete_rule=complete_rule
        self.rule=rule
        self.node_type=node_type
        self.parent=parent
        self.left_child=left_child
        self.right_child=right_child


    def _compute_complete_rule(self, first_call=True):
        """
        Compute the full rule concerning this node by recursively getting the upper tree rules
        """
        rule = ""
        if self.parent is None:
            if first_call:
                return f'{self.rule.split(" ")[0]} == {self.rule.split(" ")[0]}'
            else:
                return self.rule
        elif self.node_type=="Left":
            if (self.rule is None) or first_call:
                rule+=self.parent._compute_complete_rule(first_call=False)
            else:
                rule+=self.parent._compute_complete_rule(first_call=False) + " and " + self.rule
        else: #right node : replace parent rule with opposite rule
            parent_copy = copy.deepcopy(self.parent)
            parent_copy.rule = parent_copy.rule.replace("<=",'>').replace('==','!=')
            rule+=parent_copy._compute_complete_rule(first_call=False)
            if (self.rule is not None) and not first_call:
                rule+= " and " + self.rule
        return reduce_rule(rule)

    def set_complete_rule(self, first_call=True):
        self.complete_rule = self._compute_complete_rule(first_call=first_call)

class HierarchicalNode(Node):
    def __init__(self, number, complete_rule=None, rule=None, node_type=None, parent=None, left_child=None, right_child=None, mask=None, f1score=None, dt=None, dispersion=None, entropy=None, fisher_p=None):
        Node.__init__(self, number, complete_rule=complete_rule, rule=rule, node_type=node_type, parent=parent, left_child=left_child, right_child=right_child)
        self.mask=mask
        self.f1score=f1score
        self.dt=dt
        self.dispersion=dispersion
        self.entropy=entropy
        self.fisher_p=fisher_p
    
    def compute_f1score(self, X, zero_division=0):
        precision = self._compute_precision(X, zero_division)
        recall = self._compute_recall(X, zero_division)

        self.f1score = 2 * precision * recall / (precision + recall)

    def compute_contingency_matrix(self, X):
        if type(self.mask) is list:
            return sklearn.metrics.confusion_matrix(self.mask[0], X.eval(self.complete_rule))
        else:
            return sklearn.metrics.confusion_matrix(self.mask, X.eval(self.complete_rule))

    def _compute_precision(self, X, zero_division=0):
        if type(self.mask) is list:
            return sklearn.metrics.precision_score(self.mask[0], X.eval(self.complete_rule), zero_division=zero_division)
        else:
            return sklearn.metrics.precision_score(self.mask, X.eval(self.complete_rule), zero_division=zero_division)

    def _compute_recall(self, X, zero_division=0):
        if type(self.mask) is list:
            return sklearn.metrics.recall_score(self.mask[0], X.eval(self.complete_rule), zero_division=zero_division)
        else:
            return sklearn.metrics.recall_score(self.mask, X.eval(self.complete_rule), zero_division=zero_division)
        
    def compute_dispersion(self, explanation):
        if type(self.mask) is list:
            mean = explanation[self.mask[0]].values.mean(axis=0)
            self.dispersion = np.linalg.norm(explanation[self.mask[0]].values - mean) / self.mask[0].sum()
        else:
            mean = explanation[self.mask].values.mean(axis=0)
            self.dispersion = np.linalg.norm(explanation[self.mask].values - mean) / self.mask.sum()

    def compute_fisher_p(self):
        self.fisher_p = scipy.stats.fisher_exact(self.dt.tree_.value[1:].reshape((2,2))) if (self.dt is not None and self.dt.tree_.value.shape[0]>2) else None

class UnsupervisedNode(Node):
    def __init__(self, number, complete_rule=None, rule=None, node_type=None, parent=None, left_child=None, right_child=None, criterion=None):
        Node.__init__(self, number, complete_rule=complete_rule, rule=rule, node_type=node_type, parent=parent, left_child=left_child, right_child=right_child)
        self.criterion=criterion
