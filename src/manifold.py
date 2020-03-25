import cudf
import seaborn as sns
import matplotlib.pyplot as plt
from cuml import PCA, TSNE, UMAP

def get_manifold_2d(gdf, algo):
    return algo(n_components=2).fit_transform(gdf).set_index(gdf.index)

def plot_manifolds(manifolds, titles, labels=None, savefig=True):
    sns.set_palette("husl")
    if labels:
        fig,ax = plt.subplots(len(labels.columns),len(titles),figsize=[20,6*len(labels.columns)])
        for i,l in enumerate(labels.columns):
            for j,t in enumerate(titles):
                m = manifolds[i][j].reset_index(drop=True).to_pandas()
                mask = ~(labels[l].nans_to_nulls().isna())
                c = labels[l].loc[mask].reset_index(drop=True).to_pandas()
                c = 1 - (c - c.min())/(c.max()-c.min())
                ax[i,j].scatter(m.values[:,0],m.values[:,1],alpha=1,s=1,c=c)
                ax[i,j].set_title(t)
    else:
        fig,ax = plt.subplots(1,len(titles),figsize=[20,6])
        for i,m in enumerate(zip(manifolds, titles)):
            t = m[1]
            m = m[0].reset_index(drop=True).to_pandas()
            ax[i].scatter(m.values[:,0],m.values[:,1],alpha=1,s=2)
            ax[i].set_title(t)
            
    if savefig:
        try:
            fignum = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob('../figures/*.png')]) + 1
        except:
            fignum = 0
        fig.tight_layout();
        plt.savefig('figures/figure_{}'.format(fignum))
        
def supervised_umap(ax, manifolds, y, cutoff=50):
    x = manifolds[2].reset_index(drop=True)
    label = y['risk_score']
    c = label < cutoff

    m = UMAP(target_metric = "categorical").fit_transform(x, c)
    m = m.reset_index(drop=True).to_pandas()
    c = c.reset_index(drop=True).to_pandas()

    _=ax.scatter(m.loc[c].values[:,0],m[c].values[:,1],alpha=1,s=3,c='blue',label='Low Patient Risk Score (< {})'.format(cutoff))
    _=ax.scatter(m.loc[~c].values[:,0],m.loc[~c].values[:,1],alpha=1,s=3,c='red',label='High Patient Risk Score (> {})'.format(cutoff))
    ax.set_title('cutoff = {}%'.format(cutoff))
        
algos = {'PCA':PCA, 'tSNE':TSNE, 'UMAP':UMAP}