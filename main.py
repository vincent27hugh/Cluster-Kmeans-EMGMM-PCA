#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:35:48 2018

@author: huwei
"""
#%load_ext watermark
#import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix as sm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse 
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
#%watermark

#%watermark -p pandas,numpy,scipy,sklearn,matplotlib,seaborn

# k-means weaknesses that mixture models address directly
# code sourced from:
#   http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb

# Predefined parameters
def plot_kmeans(kmeans, X, n_clusters, rseed=2, ax=None):
    dot_size = 50
    cmap = 'viridis'
    labels = kmeans.fit_predict(X)
    
    # plot input data
    #ax = ax or plt.gca() # <-- nice trick
    fig, ax = plt.subplots(figsize=(9,7))    
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1],
               c=labels, s=dot_size, cmap=cmap, zorder=2)
    
    # plot the representation of Kmeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels==i], [center]).max() 
             for i, center in enumerate(centers)]
    
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC',edgecolor='slategrey',
                                lw=4, alpha=0.5, zorder=1))
    return  

# code sourced from:
# http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
        
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, 
                            angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    dot_size = 50
    cmap = 'viridis'
    
    fig, ax = plt.subplots(figsize=(9,7))      
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=dot_size, cmap=cmap, zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=dot_size, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, ax=ax, alpha=w * w_factor)

'''
path = '/Users/huwei/Dropbox/On_Local/' \
                   '1.Study/0_CityU_ISP/Semester A 201819/' \
                   'FB8918 Machine Learning for Business Research/' \
                   'Project 1'
os.chdir(path)
'''
wine = pd.read_csv('wine.data', \
                   names = ['Label', \
                            'Alcohol', \
                            'Malic acid', \
                            'Ash', \
                            'Alcalinity of ash', \
                            'Magnesium', \
                            'Total phenols', \
                            'Flavanoids', \
                            'Nonflavanoid phenols', \
                            'Proanthocyanins', \
                            'Color intensity', \
                            'Hue', \
                            'OD280', \
                            'Proline'])
# True labels
label = wine['Label']
del wine['Label']

# Data description
wine.dtypes


wine.describe()

# Scatter plot
sm(wine, alpha = 0.7, figsize = (18,18))
plt.show()

# Correlation Heatmap
correlation = wine.corr()
plt.subplots(figsize = (9,9))
sns.heatmap(correlation.round(2), 
            annot = True, 
            vmax = 1, 
            square = True, 
            cmap = 'RdYlGn_r')
plt.show()

# regression
sns.jointplot(x=wine.columns[5], 
              y=wine.columns[6], 
              data=wine, 
              kind="reg");
plt.show()

######################
# Normalize
scaler = preprocessing.StandardScaler()
scaler.fit(wine)
X_scaled_array = scaler.transform(wine)
winenorm = pd.DataFrame(X_scaled_array, columns = wine.columns)

########################################################################
############ K-Means
########################################################################
seed = 0
elbow = dict()
for k in range(1,11):
    estimator = KMeans(n_clusters = k,random_state=seed)
    res = estimator.fit_predict(winenorm)
    inertia = estimator.inertia_
    elbow[k] = inertia
    
elbow_df = pd.Series(elbow)
ax = elbow_df.plot(title = 'Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Inertia')
plt.plot(3,elbow_df[3],'ro')

#####
# K-Means
KM = KMeans(n_clusters = 3, random_state=seed)
res = KM.fit_predict(winenorm)

label_pred_KM = KM.labels_
print(label_pred_KM)
print('Length of labels is same as data entry', label_pred_KM.shape)

centroids_KM= KM.cluster_centers_
print(centroids_KM.shape)
print(centroids_KM)

inertia_KM = KM.inertia_
print(inertia_KM)

# Pairplot
winenorm['cluster'] = label_pred_KM.astype(str)
sns_plot = sns.pairplot(winenorm, hue = "cluster")
#sns_plot.savefig('pairplot.png')

########################################################################
##### GMM
########################################################################
aic = dict()
bic = dict()
for k in range(1,11):
	estimator = GaussianMixture(n_components = k, random_state=seed)
	res = estimator.fit(winenorm)
	ic1 = estimator.aic(winenorm)
	ic2 = estimator.bic(winenorm)
	aic[k] = ic1
	bic[k] = ic2
    
aic_df = pd.Series(aic)
bic_df = pd.Series(bic)

temp = {'AIC' : aic_df,
     'BIC' : bic_df}
ic_df = pd.DataFrame(temp)
ax = ic_df.plot(title='AIC/BIC of GMM')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('AIC/BIC')
plt.plot(3,aic_df[3],'ro')
plt.plot(3,bic_df[3],'ro')
####################
# GMM
GMM = GaussianMixture(n_components = 3, random_state=seed)
res_GMM = GMM.fit(winenorm)
res_prob_GMM = GMM.predict_proba(winenorm)
np.set_printoptions(formatter={'float_kind':'{:.3f}'.format})
print(res_prob_GMM)

weights_GMM = GMM.weights_
means_GMM = GMM.means_
covariance_GMM = GMM.covariances_
print(type(covariance_GMM))
covariance_GMM.size

label_pred_GMM = GMM.predict(winenorm)
print(label_pred_GMM)
print('Length of labels is same as data entry', label_pred_GMM.shape)

####
winenorm['cluster'] = label_pred_GMM.astype(str)

sns_plot = sns.pairplot(winenorm, hue = "cluster")

########################################################################
###### PCA
########################################################################
pca = PCA(random_state=seed)
pca.fit(winenorm)
winenorm_pca_array = pca.transform(winenorm)
winenorm_pca = pd.DataFrame(winenorm_pca_array)
print(winenorm_pca.head())
var_ratio = pca.explained_variance_ratio_
cum_var_ratio = np.cumsum(var_ratio)
print(var_ratio)
print(cum_var_ratio)
sv = pca.singular_values_
print(sv)

print(sum(var_ratio[0:5]))

# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
plt.figure(figsize=(10, 5))
plt.bar(range(len(var_ratio)), 
        var_ratio, 
        alpha=0.3333, 
        align='center', 
        label='individual explained variance', 
        color = 'g')
plt.step(range(len(cum_var_ratio)),
         cum_var_ratio, 
         where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

#################################
# Plot PCA1~PCA2
pca = PCA(n_components=2, random_state=seed)
pca.fit(winenorm)
winenorm_pca_array = pca.transform(winenorm)
winenorm_PCA = pd.DataFrame(winenorm_pca_array)
plt.scatter(x=winenorm_PCA.iloc[:,0], 
            y=winenorm_PCA.iloc[:,1], 
            alpha = 0.7)
plt.ylabel('PCA2')
plt.xlabel('PCA1')
plt.show()

#######
KM = KMeans(n_clusters = 3, random_state=seed)
plot_kmeans(KM, winenorm_PCA.as_matrix(),n_clusters=3)
plt.ylabel('PCA2')
plt.xlabel('PCA1')
plt.title('Clusters of K-Means in two PCAs', fontsize=18, fontweight='demi')
###

GMM = GaussianMixture(n_components = 3, random_state=seed)
plot_gmm(GMM, winenorm_PCA.as_matrix())
plt.ylabel('PCA2')
plt.xlabel('PCA1')
plt.title('Clusters of GMM in two PCAs', fontsize=18, fontweight='demi')
##########################################
# we slecte six components
pca = PCA(n_components = 6, random_state=seed)
pca.fit(winenorm)
winenorm_pca_array = pca.transform(winenorm)
winenorm_PCA = pd.DataFrame(winenorm_pca_array)
print(winenorm_pca)
var_ratio = pca.explained_variance_ratio_
print(var_ratio)
sv = pca.singular_values_
print(sv)
print(sum(var_ratio[0:5]))
# K-Means + PCA
KM = KMeans(n_clusters = 3, random_state=seed)
res = KM.fit(winenorm_PCA)
label_pred_KM_PCA = KM.predict(winenorm_PCA)

# GMM + PCA
GMM = GaussianMixture(n_components = 3, random_state=seed)
res_GMM = GMM.fit(winenorm_PCA)
label_pred_GMM_PCA = GMM.predict(winenorm_PCA)

########################################################################
# Evaluation
from sklearn import metrics

sh_score_KM = metrics.silhouette_score(winenorm, label_pred_KM)
print(sh_score_KM)
sh_scores_KM = metrics.silhouette_samples(winenorm, label_pred_KM)
sns.distplot(sh_scores_KM);

sh_score_KM_PCA = metrics.silhouette_score(winenorm, label_pred_KM_PCA)
print(sh_score_KM_PCA)
sh_scores_KM_PCA = metrics.silhouette_samples(winenorm, label_pred_KM_PCA)
sns.distplot(sh_scores_KM_PCA);

sh_score_GMM = metrics.silhouette_score(winenorm, label_pred_GMM)
print(sh_score_GMM)
sh_scores_GMM = metrics.silhouette_samples(winenorm, label_pred_GMM)
sns.distplot(sh_scores_GMM);

sh_score_GMM_PCA = metrics.silhouette_score(winenorm, label_pred_GMM_PCA)
print(sh_score_GMM_PCA)
sh_scores_GMM_PCA = metrics.silhouette_samples(winenorm, label_pred_GMM_PCA)
sns.distplot(sh_scores_GMM_PCA);

###
from sklearn.metrics.cluster import adjusted_rand_score

# 
ar_score_KM = adjusted_rand_score(label, label_pred_KM)
print(ar_score_KM)

ar_score_KM_PCA = adjusted_rand_score(label, label_pred_KM_PCA)
print(ar_score_KM_PCA)

ar_score_GMM = adjusted_rand_score(label, label_pred_GMM)
print(ar_score_GMM)

ar_score_GMM_PCA = adjusted_rand_score(label, label_pred_GMM_PCA)
print(ar_score_GMM_PCA)
