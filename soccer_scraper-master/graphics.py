import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import geopandas

'''
FIXED = note index in cluster map
'''

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))


def gen_fig(model, res=.1, college='Stanford', position='Defender'):
    cluster_map = {c: model.predict(college, {'cluster': c, 'Position': position}, relative=True)[0] for c in
                   model.data.cluster.unique()}

    X = model.data[['longitude', 'latitude']]
    (lon_min, lon_max), (lat_min, lat_max) = list(zip(X.min().values, X.max().values))
    xx, yy = np.mgrid[lon_min:lon_max:res, lat_min:lat_max:res]
    Z = np.array([cluster_map[v] for v in model.kmeans.predict(np.c_[xx.ravel(), yy.ravel()])])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape).T

    fig, ax = plt.subplots(figsize=(15, 11))
    ax1 = plt.imshow(Z, interpolation='bicubic',
                     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                     aspect='auto', origin='lower')
    plt.colorbar(ax1)

    centers = model.kmeans.cluster_centers_
    X, Y = list(zip(*centers))

    ax2 = sns.scatterplot(y='latitude', x='longitude', hue='cluster', data=model.data, palette='RdBu', legend=False)
    annotations = []
    for i, (x, y) in enumerate(centers):
        anno = ax2.annotate(i, (X[i], Y[i]), fontsize=20, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='red'))
        annotations.append(anno)

    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)

    adjust_text(
        annotations)  # , only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    plt.title('Geographic Dist of Probs')

    return fig


def cluster_map(model):
    fig, ax = plt.subplots(figsize=(15, 11))
    base = world.plot(ax=ax, color='green', edgecolor='black')
    _ = sns.scatterplot(ax=base, y='latitude', x='longitude', hue='cluster', data=model.data, palette='inferno',
                        legend=False)

    centers = model.kmeans.cluster_centers_
    X, Y = list(zip(*centers))
    annotations = []
    for i, (x, y) in enumerate(centers):
        anno = base.annotate(i, (X[i], Y[i]), fontsize=20, fontweight='bold',
                             bbox=dict(facecolor=(1, 1, 1, .6)))
        annotations.append(anno)
    adjust_text(annotations)
    plt.title('Cluster Map')
    return fig


