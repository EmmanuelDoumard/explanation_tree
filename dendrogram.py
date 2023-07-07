import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
import plotly
import plotly.graph_objects as go
import shap
from plotly.figure_factory import create_dendrogram
import plotly.figure_factory._dendrogram as original_dendrogram
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import scipy.stats as stats
from statistics import mean, stdev
import math

class ExplanationDendrogram:
    def __init__(self, explanation, X, metric="euclidean", method="ward", optimal_ordering=True, link=None, p=30, truncate_mode=None):
        self.explanation=explanation
        self.X=X
        self.metric=metric
        self.method=method
        self.optimal_ordering=optimal_ordering
        self.link=link
        self.dend=None
        self.cluster_list=None
        self.p=p
        self.truncate_mode=truncate_mode


    def compute_link(self):
        if self.link is not None:
            print("Linkage has already been computed as link")
        else:
            self.link = shc.linkage(self.explanation.values, metric=self.metric, method=self.method, optimal_ordering=self.optimal_ordering)
    
    def compute_dend(self):
        if self.link is None:
            self.compute_link()
        self.dend = shc.dendrogram(self.link, p=self.p, truncate_mode=self.truncate_mode, no_plot=True)

    def plot_dend(self, figsize=(15,6), savefig=None, color_threshold=None, orientation='top', labels=None,  count_sort=False, distance_sort=False, show_leaf_counts=True, no_labels=False, leaf_font_size=None, leaf_rotation=None, leaf_label_func=None, show_contracted=False, link_color_func=None, ax=None, above_threshold_color='C0'):
        if self.dend is None:
            self.compute_dend()
        plt.figure(figsize=figsize)
        shc.dendrogram(self.link, p=self.p, truncate_mode=self.truncate_mode, color_threshold=color_threshold, orientation=orientation, labels=labels,  count_sort=count_sort, distance_sort=distance_sort, show_leaf_counts=show_leaf_counts, no_labels=no_labels, leaf_font_size=leaf_font_size, leaf_rotation=leaf_rotation, leaf_label_func=leaf_label_func, show_contracted=show_contracted, link_color_func=link_color_func, ax=ax, above_threshold_color=above_threshold_color)
        if savefig: plt.savefig("dendro.png")
        plt.show()
        

    def annotate_dendrogram(self, fig, link_trunc=None):
        if link_trunc is not None:
            offset = len(self.link)-len(link_trunc) #Sert en cas de troncature
        else:
            offset = 0
        if link_trunc is not None:
            Z=link_trunc
        else:
            Z=self.link
        for i in range(len(Z)):
            node = np.where(np.array(self.dend['dcoord'])[:,1] == Z[i][2])[0][0]
            fig.add_annotation({"text":str(i+1+offset),"x": 0.5 * sum(self.dend["icoord"][node][1:3]),"y":self.dend["dcoord"][node][1], "showarrow":False,"yanchor":"top"})

    def _styling_non_equal(self, s, cell_color, X, datatype, cluster1, cluster2):
        ps = [0.05, 0.01, 0.001]
        colors = ['#e6ffe6;','#78ff00;','#01cb48;']
        s.set_table_styles([{'selector': '.false', 'props': 'background-color: #ffe6e6;'}]+[{'selector': '.'+'a'*(i+1), 'props': f'background-color: {colors[i]}'} for i in range(len(ps))], overwrite=False)
        for col in X.columns:
            cell_color.loc[(col,'mean'),datatype] = 'false'
            for i,p in enumerate(ps):
                if stats.ttest_ind(X.loc[X['cluster']==cluster1,col],X.loc[X['cluster']==cluster2,col], equal_var=False).pvalue < p:
                    cell_color.loc[(col,'mean'),datatype] = 'a'*(i+1)

    def _styling_cohen_d(self, s, cell_color, X, datatype, cluster1, cluster2):
        cohen_t = [0.2, 0.5, 0.8]
        colors = ['#e6ffe6;','#78ff00;','#01cb48;']
        s.set_table_styles([{'selector': '.false', 'props': 'background-color: #ffe6e6;'}]+[{'selector': '.'+'a'*(i+1), 'props': f'background-color: {colors[i]}'} for i in range(len(cohen_t))], overwrite=False)
        for col in X.columns:
            cell_color.loc[(col,'mean'),datatype] = 'false'
            for i,p in enumerate(cohen_t):
                c0=X.loc[X['cluster']==cluster1,col]
                c1=X.loc[X['cluster']==cluster2,col]
                if (math.sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2)) ==0:
                    cohens_d=0
                else:
                    cohens_d = np.abs(mean(c0) - mean(c1)) / (math.sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
                if cohens_d > p:
                    cell_color.loc[(col,'mean'),datatype] = 'a'*(i+1)

                
    def gen_cluster_list(self):
        cluster_list = {i:[i] for i in range(len(self.link)+1)}
        for i in range(0,len(self.link)):
            cluster_list[i+1 + len(self.link)] = cluster_list[int(self.link[i][0])] + cluster_list[int(self.link[i][1])]
        self.cluster_list = cluster_list

    def plot_widgets_interactive_dendrogram(self, orientation="bottom"):
        """
        Wrapper for the dendrogram plotting functions of scipy and plotly.
        Creates an ipywidget to interact with the created dendrogram.
        """
        ### MONKEY PATCHING ###
        p2 = self.p
        truncate_mode2 = self.truncate_mode
        def modified_dendrogram_traces(self, X, colorscale, distfun, linkagefun, hovertext, color_threshold):
            """
            Calculates all the elements needed for plotting a dendrogram.

            :param (ndarray) X: Matrix of observations as array of arrays
            :param (list) colorscale: Color scale for dendrogram tree clusters
            :param (function) distfun: Function to compute the pairwise distance
                                    from the observations
            :param (function) linkagefun: Function to compute the linkage matrix
                                        from the pairwise distances
            :param (list) hovertext: List of hovertext for constituent traces of dendrogram
            :rtype (tuple): Contains all the traces in the following order:
                (a) trace_list: List of Plotly trace objects for dendrogram tree
                (b) icoord: All X points of the dendrogram tree as array of arrays
                    with length 4
                (c) dcoord: All Y points of the dendrogram tree as array of arrays
                    with length 4
                (d) ordered_labels: leaf labels in the order they are going to
                    appear on the plot
                (e) P['leaves']: left-to-right traversal of the leaves

            """
            import plotly
            from plotly import exceptions, optional_imports
            np = optional_imports.get_module("numpy")
            scp = optional_imports.get_module("scipy")
            sch = optional_imports.get_module("scipy.cluster.hierarchy")
            scs = optional_imports.get_module("scipy.spatial")
            sch = optional_imports.get_module("scipy.cluster.hierarchy")
            d = distfun(X)
            Z = linkagefun(d)
            P = sch.dendrogram(
                Z,
                orientation=self.orientation,
                labels=self.labels,
                no_plot=True,
                color_threshold=color_threshold,
                truncate_mode = truncate_mode2,
                p = p2
            )

            icoord = np.array(P["icoord"])
            dcoord = np.array(P["dcoord"])
            ordered_labels = np.array(P["ivl"])
            color_list = np.array(P["color_list"])
            colors = self.get_color_dict(colorscale)

            trace_list = []

            for i in range(len(icoord)):
                # xs and ys are arrays of 4 points that make up the '∩' shapes
                # of the dendrogram tree
                if self.orientation in ["top", "bottom"]:
                    xs = icoord[i]
                else:
                    xs = dcoord[i]

                if self.orientation in ["top", "bottom"]:
                    ys = dcoord[i]
                else:
                    ys = icoord[i]
                color_key = color_list[i]
                hovertext_label = None
                if hovertext:
                    hovertext_label = hovertext[i]
                trace = dict(
                    type="scatter",
                    x=np.multiply(self.sign[self.xaxis], xs),
                    y=np.multiply(self.sign[self.yaxis], ys),
                    mode="lines",
                    marker=dict(color=colors[color_key]),
                    text=hovertext_label,
                    hoverinfo="text",
                )

                try:
                    x_index = int(self.xaxis[-1])
                except ValueError:
                    x_index = ""

                try:
                    y_index = int(self.yaxis[-1])
                except ValueError:
                    y_index = ""

                trace["xaxis"] = "x" + x_index
                trace["yaxis"] = "y" + y_index

                trace_list.append(trace)

            return trace_list, icoord, dcoord, ordered_labels, P["leaves"]

        original_dendrogram._Dendrogram.get_dendrogram_traces = modified_dendrogram_traces
        
        ### END OF MONKEY PATCHING ###

        #Create dendrogram
        if self.dend is None:
            if self.link is None:
                self.compute_link()
            self.compute_dend()
        
        if self.truncate_mode is None:
            link_trunc=None
        else:
            link_trunc = np.array([self.link[np.where(self.link==self.dend["dcoord"][i][1])[0][0]] for i in range(len(self.dend["dcoord"]))]) #truncate linkage
            link_trunc = link_trunc[link_trunc[:,2].argsort()] #re-order truncated linkage by impurity
        
        fig = create_dendrogram(self.explanation.values, orientation=orientation, linkagefun=lambda x:self.link, hovertext=["Cluster " + str(np.where(self.link==self.dend["dcoord"][i][1])[0][0] +1) for i in range(len(self.dend["dcoord"]))])
        fig.update_layout(width=1200,height=600,
                        hoverlabel={"align":"auto"},
                        modebar={"add":[["zoom2d"],["pan2d"],["select2d"],["zoomIn2d"],["zoomOut2d"],["autoScale2d"],["resetScale2d"]]})
        self.annotate_dendrogram(fig, link_trunc)
        f = go.FigureWidget(fig)
        
        self.gen_cluster_list()    
        out21 = widgets.Output(width='50%')
        out21.layout.border = '1px solid'
        out22 = widgets.Output(width='50%')
        out22.layout.border = '1px solid'
        out3 = widgets.Output()
        decision21 = widgets.Image(format="png", layout={'border':'1px solid'})
        decision22 = widgets.Image(format="png", layout={'border':'1px solid'})
        
        ### DEF WIDGET CALLBACK ###
        def show_members(trace, points, selector, cluster_list=self.cluster_list, Z=self.link, X=self.X, explanation=self.explanation):
            if trace.line.width > 2:
                trace.line.width -=2
            if len(points.point_inds)>0:
                trace.line.width+=2
                members_list = cluster_list[np.where(Z[:,2] == trace['y'][2])[0][0]+len(cluster_list)//2+1]
                if (points.ys[0]==0) and (len(members_list)==2): #leaf
                    if points.xs[0]==trace.x[0]: #left leaf
                        members_list=members_list[0]
                    else: #right leaf
                        members_list=members_list[-1]
                dpi=80
                plt.figure()
                shap.plots.decision(explanation.base_values[0],
                                    explanation.values[members_list], 
                                    features=X.iloc[members_list], 
                                    title="Cluster " + str(np.where(Z[:,2] == trace['y'][2])[0][0] + 1),
                                    show=False,
                                    auto_size_plot=True,
                                    ignore_warnings=True
                                    )
                plt.plot(np.concatenate(([explanation.base_values[0]],explanation.base_values[0]+explanation.values[members_list].mean(axis=0)[np.abs(explanation.values[members_list]).mean(axis=0).argsort()].cumsum()[-min(X.shape[1],20):])),
                        np.arange(0,min(X.shape[1],20)+1),
                        color="black",
                        linewidth=3
                        )
                plt.savefig("decision_members.png",bbox_inches="tight")
                file = open("decision_members.png","rb")
                image = file.read()
                im = cv2.imread("decision_members.png")

                if selector.ctrl:
                    decision = decision22
                    out=out22
                    otherout=out21
                else:
                    decision = decision21
                    out=out21
                    otherout=out22

                decision.value=image
                decision.height=im.shape[0]
                decision.width=im.shape[1]

                out.clear_output()
                out.layout.width = str(decision.width)+"px"
                with out:
                    print(np.where(Z[:,2] == trace['y'][2])[0][0] + 1)
                    #print(members_list)
                    print("Details : ")
                    display(X.iloc[members_list].describe().loc[["count","min","mean","50%","max","std"]].style.format(precision=2))
                    cluster1 = int(np.where(Z[:,2] == trace['y'][2])[0][0] + 1)

                if len(otherout.outputs)>0:
                    cluster2 = int(otherout.outputs[0]["text"].split("\n")[0])
                    X2 = X.copy()
                    X2["cluster"]=np.nan
                    X2.loc[X2.iloc[members_list].index,"cluster"]=cluster1
                    other_members_list = self.cluster_list[cluster2+len(cluster_list)//2]
                    X2.loc[X2.iloc[other_members_list].index,"cluster"]=cluster2

                    expl2 = pd.DataFrame(explanation.values, columns=X.columns, index=X.index)
                    expl2["cluster"]=X2["cluster"]
                    out3.clear_output()
                    
                    with out3:
                        df_to_display=X2.groupby("cluster").describe().drop(["count","25%","75%"],axis=1,level=1)
                        df_to_display.loc[cluster1,"count"] = len(members_list)
                        df_to_display.loc[cluster2,"count"] = len(other_members_list)
                        df_to_display.index=df_to_display.index.astype(int)
                        df_to_display = df_to_display.loc[:,["count"]+df_to_display.drop("count",axis=1).columns.get_level_values(0).unique().to_list()] #put count at the beginning
                        
                        df_to_concat = expl2.groupby("cluster").describe().drop(["count","25%","75%"],axis=1,level=1)
                        df_to_concat.loc[cluster1,"count"] = len(members_list)
                        df_to_concat.loc[cluster2,"count"] = len(other_members_list)
                        df_to_concat.index=df_to_concat.index.astype(int)
                        df_to_concat = df_to_concat.loc[:,["count"]+df_to_concat.drop("count",axis=1).columns.get_level_values(0).unique().to_list()]
                        
                        df_final = pd.concat([df_to_display.T,df_to_concat.T],axis=1)
                        df_final.columns = pd.MultiIndex.from_tuples([("Data",df_to_display.T.columns[0]),
                                                                    ("Data",df_to_display.T.columns[1]),
                                                                    ("Explanations",df_to_display.T.columns[0]),
                                                                    ("Explanations",df_to_display.T.columns[1])])
                        
                        s = df_final.style.format(precision=2, subset=["Data"]).format(precision=4, subset=["Explanations"])
                        s.set_table_styles([  # create internal CSS classes
                            {'selector': 'th.col_heading', 'props': 'text-align: center;'},
                            {'selector': 'tr:hover', 'props': [('font-weight','bolder')]}
                        ], overwrite=False)
                        print(df_to_concat.T.columns[0])
                        s.set_table_styles({
                            ('Explanations', df_to_concat.T.columns[0]): [{'selector': 'th', 'props': 'border-left: 1px solid #000066'},
                                                            {'selector': 'td', 'props': 'border-left: 1px solid #000066'}]
                        }, overwrite=False, axis=0)
                        s.set_table_styles({
                            (col, 'mean'): [{'selector': 'td', 'props': 'border-top: 1px solid #000066'},
                                            {'selector': 'th', 'props': 'border-top: 1px solid #000066'}] for col in df_final.index.get_level_values(0).unique().to_list()
                        }, overwrite=False, axis=1)
                        
                        cell_color = pd.DataFrame(np.zeros(df_final.shape), index=df_final.index, columns=df_final.columns)
                        #cell_color[:] = 'false'
                        
                        #self._styling_non_equal(s, cell_color, X2, 'Data', cluster1, cluster2)
                        #self._styling_non_equal(s, cell_color, expl2, 'Explanations', cluster1, cluster2)             
                        self._styling_cohen_d(s, cell_color, X2, 'Data', cluster1, cluster2)            
                        self._styling_cohen_d(s, cell_color, expl2, 'Explanations', cluster1, cluster2)        
                        
                        s.set_td_classes(cell_color)
                        display(s)
        for i in range(len(f.data)):
            f.data[i].on_click(show_members)
            f.data[i].on_selection(show_members)
            
        return widgets.VBox([f,widgets.HBox([widgets.VBox([decision21, out21]),widgets.VBox([decision22, out22])]), out3])