import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import shap.plots

def draw_dendro(values, p=30, truncate_mode=None, figsize=(18,12), title="Dendrogram", metric="euclidean", method="ward", optimal_ordering=True, no_labels=False, **kwargs):
    """
    Draw the dendrogram associated with the agglomerative clustering performed on the input data
    
    Parameters
    ----------
    values : ndarray, list
        array of values to perform the agglomerative clustering on
        
    Returns
    -------
    ndarray, dict
        The linkage array computed by the agglomerative clustering method
        The dendrogram dict for plot purposes
    """   
    
    plt.figure(figsize=figsize)  
    plt.title(title)  
    link = shc.linkage(values, metric=metric, method=method, optimal_ordering=optimal_ordering)
    dend = shc.dendrogram(link, p=p, truncate_mode=truncate_mode, no_labels=no_labels, **kwargs)
    return link, dend

def lmethod(link, limit=None, draw=False):
    """
    Performs the L method on the linkage matrix to find the best number of clusters
    
    Parameters
    ----------
    link : ndarray
        Linkage array computed by the agglomerative clustering
        
    limit : int, optionnal
        Maximum number of clusters in the data
        If none, the maximum number of clusters is the number of instances (unit clustering)
        
    draw : boolean
        If True, plots the evaluation graph as well as the two  best regression lines made to fit the L method
        
    Returns
    -------
    int
        The number of clusters calculated by the L method
    """
    
    if limit is None:
        limit = link.shape[0]
    elif limit > link.shape[0]:
        limit = link.shape[0]
    elif limit < 20:
        limit = 20
    eval_graph = link[::-1,2][:limit]  #Takes the limit number of mergings from the end and reverse the list
    min_error = np.inf
    chat = 0
    for c in range(1,limit-1):
        Lc = sklearn.linear_model.LinearRegression()
        Lc.fit(np.arange(0,c).reshape(-1,1),eval_graph[:c])
        Rc = sklearn.linear_model.LinearRegression()
        Rc.fit(np.arange(c,limit).reshape(-1,1), eval_graph[c:])
        
        RMSELc = sklearn.metrics.mean_squared_error(eval_graph[:c],Lc.predict(np.arange(0,c).reshape(-1,1)), squared=False)
        RMSERc = sklearn.metrics.mean_squared_error(eval_graph[c:],Rc.predict(np.arange(c,limit).reshape(-1,1)), squared=False)
        
        RMSEc = ((c-1)/(limit-1))*RMSELc + ((limit-c)/(limit-1))*RMSERc
        
        # print(c)
        # print("RMSELc : ",RMSELc)
        # print("RMSERc : ",RMSERc)
        # print("RMSEc : ", RMSEc)
        
        if RMSEc < min_error:
            best_Lc = Lc
            best_Rc = Rc
            min_error = RMSEc
            chat = c
    if draw:
        plt.figure(figsize=(16,6))
        plt.scatter(range(0,limit),eval_graph)
        plt.plot(range(0,chat),best_Lc.intercept_+best_Lc.coef_[0]*range(0,chat))
        plt.plot(range(chat,limit),best_Rc.intercept_+best_Rc.coef_[0]*range(chat,limit))
        plt.show()
    return chat+1

def iterative_refinement(data=None, link=None, draw=False):
    """
    Iteratively refines the limit of the L method to compute the number of clusters based on the linkage matrix
    
    Parameters
    ----------
    link : ndarray
        Linkage array computed by the agglomerative clustering
        
    draw : boolean
        If True, plots each evaluation graph as well as the two best regression lines made to fit the L method
        at each refinement
        
    Returns
    -------
    int
        The number of clusters calculated by the method
    """
    if (data is None) and (link is None):
        raise ValueError("Specify at least either the data or the linkage matrix")
    elif (data is not None) and (link is None):
        link = shc.linkage(data, metric="euclidean", method="ward", optimal_ordering=True)
    limit = None
    lastKnee = link.shape[0]+1
    currentKnee = link.shape[0]
    
    while (currentKnee < lastKnee):
        lastKnee = currentKnee
        currentKnee = lmethod(link, limit, draw)
        limit = currentKnee*2
    return currentKnee

def generate_correct_explanation(explanation=None,shap_values=None,base_values=None,X=None,look_at=1):
    """
    Recreate a shap explanation object to fit the same convention
    
    """
    
    if explanation is None:
        if (shap_values is None) | (base_values is None) | (X is None):
            raise Exception("If you pass no explanation, you need to pass shap_values, base_values and X to construct an Explanation object")
            
    if shap_values is None:
        if len(np.array(explanation.values).shape) == 3:
            shap_values = explanation.values[:,:,look_at]
        else:
            shap_values = explanation.values
    if base_values is None:
        if len(np.array(explanation.base_values).shape) == 2:
            base_values = explanation.base_values[:,look_at]
        else:
            base_values = explanation.base_values
    if X is None:
        X = pd.DataFrame(explanation.data,columns=explanation.feature_names)
        
    correct_explanation = shap.Explanation(shap_values,
                                         base_values=base_values,
                                         data=X.values,
                                         feature_names=X.columns.to_list())

    return correct_explanation

def clustering_explanations(explanation):
    """
    Perform agglomerative clustering on the explanations and show the different clusters
    
    Parameters
    ----------
    explanation: shap.Explanation
        Explanation object containing influences, feature names, etc.
        
    Returns
    -------
    ndarray
        1D array of size equal to the number of instances in the explanation
    """
    
    link, dend = draw_dendro(explanation.values)
    
    knee = iterative_refinement(link=link)
    cluster_labels = shc.fcluster(link, knee, criterion="maxclust")
    
    for clust in np.unique(cluster_labels):
        plt.figure()
        if len(explanation.values.shape)>2:
            shap.plots.beeswarm(explanation[cluster_labels==clust,:,0], show=False)
        else:
            shap.plots.beeswarm(explanation[cluster_labels==clust,:], show=False)
        plt.title("Cluster number "+clust)
        plt.show()
    
    return cluster_labels
