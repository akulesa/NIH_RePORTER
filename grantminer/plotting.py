from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def pca_curve(vectors):
    """ Plot the amount of variance explained for each additional PCA component.
    Inputs:
        - vectors, n x m numpy array, n = number of vectors, m = dimensionality
    Outputs:
        - Bokeh figure
    """
    m = vectors.shape[1]
    pca = PCA(n_components=m)
    pca_result = pca.fit_transform(vectors)

    hover = HoverTool(
    tooltips=[
            ("features","$x"),
            ("explained variance", "$y"),
        ]
    )

    p = figure(plot_width=500,plot_height=300,tools=[hover])
    p.line(x = np.arange(m),y = np.cumsum(pca.explained_variance_ratio_))

    p.xaxis.axis_label = "Features"
    p.yaxis.axis_label = "% Variance Explained"
    return p
