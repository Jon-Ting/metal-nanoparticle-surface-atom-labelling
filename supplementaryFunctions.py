# Python script containing functions used for data preprocessing of dataframes generated from atom-wise feature extraction
# Author: Jonathan Yik Chang Ting
# Date: 15/1/2022

import sys
import warnings
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import numpy as np
import modin.pandas as modpd
import pandas as pd
import seaborn as sns
import sklearn
from statsmodels.graphics.gofplots import qqplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from scipy.stats import probplot, shapiro, normaltest, anderson, skew, kurtosis


# Set up environment
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('plotting.backend', 'plotly')
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(color_codes=True)
sns.set(font_scale=1.2)
warnings.filterwarnings("ignore")
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
# rcParams['figure.figsize'] = [8, 5]
# rcParams['figure.dpi'] = 80
rcParams['figure.autolayout'] = True
rcParams['font.style'] = 'normal'
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
randomSeed = 777
corrMethod = "spearman"
varThreshs, corrThreshs = [0.01, 0.03, 0.05], [0.90, 0.95, 0.99]


# Author: mot032, Date: Fri Jan  4 14:57:13 2019
class Atom(object):
    def __init__(self):
        self.mass = 1
        self.xnew = None
        self.ynew = None
        self.znew = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.vx_half = 0.0
        self.vy_half = 0.0
        self.vz_half = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
        self.charge = 0.0
        self.partial_charge = 0.0
        self.ID = None
        self.Type = None
        self.TypeID = None
        self.FF_Type = None
        self.bonded_neighbour = []
        self.coord = None
        self.visited = False
        

def dfMassage(df, interactive=True):
    """
    Handle missing values, drop duplicates in input DataFrame, one-hot encode certain features, and turn columns datatype into numeric
    input:
        df = input DataFrame
        interactive = Boolean indicator to decide usage of function (display/print) for DataFrame inspection
    output:
        dfNew = processed DataFrame
    """
    # Handle missing values
    if df.isna().any().any():
        print("Missing entries exist!")
        missingNum = df.isna().sum()
        missingNum[missingNum > 0].sort_values(ascending=False)
        print("Missing rows in each column/feature:", missingNum)
        # df.dropna(axis=0, how='any', thresh=None, subset=["csm", "molRMS"], inplace=True)
        # df.replace({"csm": {np.nan: dfAllEnc["csm"].max()}, 
        #                "molRMS": {np.nan: dfAllEnc["molRMS"].max()}, 
        #                "Ixx": {np.nan: dfAllEnc["Ixx"].mean()}, # Check if average is appropriate
        #                "Iyy": {np.nan: dfAllEnc["Iyy"].mean()}, 
        #                "Izz": {np.nan: dfAllEnc["Izz"].mean()}, 
        #                "E": {0.0: 1.0}}, 
        #                inplace=True, regex=False, limit=None, method="pad")
    # df.replace(to_replace=np.inf, value=1000, inplace=True, regex=False, limit=None, method="pad")  # Check if 1000 is appropriate

    # Drop duplicates
    if df.duplicated().any():
        print("Duplicate entries exists!")
        print("Number of rows before dropping duplicates: ", len(df))
        df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False);
        print("Number of rows after dropping duplicates: ", len(df))
        
    # One-hot encode features
    dummyEle = pd.get_dummies(df['ele'], prefix='ele')
    dfNew = pd.merge(left=df, right=dummyEle, left_index=True, right_index=True)
    
    # To numeric
    dfNew = dfNew.astype({"x": float, "y": float, "z": float, 
                    "xNorm": float, "yNorm": float, "zNorm": float, 
                    "Ixx": float, "Iyy": float, "Izz": float, 
                    "degenDeg": float})
    
    print("\nMissing values handled, duplicates dropped, one-hot encoded categorical features, feature columns numericised:")
    display(dfNew.sample(5)) if interactive else print(dfNew.sample(5))
    return dfNew


def dfDropUnusedCols(df):
    """
    Drop unused feature columns (could vary depending on datasets e.g. mono/multi-metallic)
    input:
        df = input DataFrame
    output:
        dfNew = processed DataFrame
    """
    # "x", "y", "z" redundant (included normalised columns)
    # "pg" could only be one-hot encoded when all possibilities are recorded
    # "csm", "molRMS" undefined when "pg" is missing
    # inclusion of "E", "C8", etc. makes clustering difficult
    pgIdx = list(df.columns).index("pg")
    surfIdx = list(df.columns).index("surf")
    cols2drop = ["x", "y", "z", "ele"] + list(df.columns)[pgIdx:surfIdx]
    dfNew = df.drop(labels=cols2drop, axis=1, index=None, columns=None, level=None, inplace=False, errors="raise")
    dfNew.rename(columns={'xNorm': 'x', 'yNorm': 'y', 'zNorm': 'z'}, inplace=True)
    return dfNew


def mapGauss(df):
    """
    Map distribution of features to Gaussian distributions via power tranform (parametric, monotonic transformations)
        Box-Cox tranformation (only applicable to strictly positive data)
        Yeo-Johnson transformation
        Quantile transformation
    input: 
        df = DataFrame with columns being features
    output:
        dfYJTMap = DataFrame after Yeo-Johnson transformation
        dfQTMap = DataFrame after quantile transformation
    """ 
    YJT = PowerTransformer(method='yeo-johnson', standardize=False, copy=True)
    QT = QuantileTransformer(n_quantiles=1000, output_distribution='normal', ignore_implicit_zeros=False, subsample=100000, random_state=randomSeed, copy=True)
    YJTmappedArr = YJT.fit_transform(df)
    QTmappedArr = QT.fit_transform(df)
    dfYJTMap = pd.concat(objs=[pd.DataFrame(YJTmappedArr, index=df.index, columns=df.columns)], axis=1)
    dfQTMap = pd.concat(objs=[pd.DataFrame(QTmappedArr, index=df.index, columns=df.columns)], axis=1)
    return dfYJTMap, dfQTMap


def normalTest(df, alpha=0.05, verbose=False):
    """
    Assess whether the normality assumption holds for each feature.
        Shapiro-Wilk Test quantifies how likely the data is drawn from Gaussian distribution. (W accurate for N > 5000 but not p)
        D'Agostino's K^2 Test calculates summary statistics from data to quantify deviation from Gaussian distribution (statistics = sum of square of skewtest's and kurtosistest's z-score)
        Anderson-Darling Test evaluates whether a samples comes from one of among many known samples
    input: 
        df = DataFrame with columns being features
        alpha = significance level
        verbose = Boolean indicator for printing output
    output:
        vioList = list of features that violate the normality assumption
        normList = list of features that satisfy the normality assumption
    """ 
    vioList, normList = [], []
    for feat in df.columns:
        # Statistical checks (Quantification)
        if verbose: print("\nFeature: {0}".format(feat))
        xArr = df[feat]
        Wstat, WpVal = shapiro(x=xArr)
        Dstat, DpVal = normaltest(a=xArr, axis=None, nan_policy='propagate')
        Astat, AcritVals, AsigLev = anderson(x=xArr, dist='norm')
        if WpVal < alpha or DpVal < alpha or Astat > AcritVals.any():
            if verbose:
                print("  Shapiro-Wilk Test p-value: {0:.3f}\n  D'Agostino's K^2 Test p-value: {1:.3f}".format(WpVal, DpVal))
                print("  Anderson-Barling Test statistics: {0:.3f}".format(Astat))
                for (i, sigLev) in enumerate(AsigLev): print("    Significance level: {0:.3f}, Critical value: {1:.3f}".format(sigLev, AcritVals[i]))
            if Astat > AcritVals.all():
                vioList.append(feat)
                if verbose: print("Statistically significant at all significance level, normality assumption violated!")
            else:
                normList.append(feat)
                if verbose: print("Hypothesis couldn't be rejected!")
    
            # Visual checks (Qualification)
            if verbose: 
                plt.figure(figsize=(12, 5));
                plt.subplot(121);
                sns.histplot(data=df, x=feat, kde=True);
                plt.subplot(122);
                probplot(x=xArr, dist="norm", fit=True, plot=plt, rvalue=True);
                plt.show();
    
    print("  Features checked: ", list(df.columns))
    print("  Normality violated by: ", vioList)
    return vioList, normList


def dfNorm(df, alpha=0.05, verbose=False, interactive=True):
    """
    Scale the dataset after checking normality assumption
    input:
        df = input DataFrame
        alpha = significance level
        verbose = Boolean indicator for output printing
        interactive = Boolean indicator to decide usage of function (display/print) for DataFrame inspection
    output:
        dfScaled = scaled DataFrame
    """
    print("\nCheck normality assumptions prior to transfomation:")
    vioList, normList = normalTest(df, alpha=alpha, verbose=False)
    
    # Transform the distributions of numerical features to Gaussian distributions and perform the same check
    dfYJTMap, dfQTMap = mapGauss(df)
    print("\nCheck normality assumptions after Yeo-Johnson transformation:")
    vioList1, normList1 = normalTest(dfYJTMap, alpha=alpha, verbose=False)
    print("\nCheck normality assumptions after quantile tranformation:")
    vioList2, normList2 = normalTest(dfQTMap, alpha=alpha, verbose=False)
    
    # Normalisation/Standardisation/Robust scaling
    scalerType = "minMax" if len(vioList1) > 0 or len(vioList2) > 0 else "stand"  # Robust scaling not considered for regularly ordered nanoparticles
    scaler, dfScaled = dfScale(df, catCols=["surf"], scaler=scalerType)
    print("\nScaling the data, scaler: {0}".format(scalerType))
    display(dfScaled.sample(5)) if interactive else print(dfScaled.sample(5))
    return dfScaled


def dfScale(df, catCols=None, scaler="minMax"):
    """
    Scale input feature DataFrame using various sklearn scalers
    input:
        df = DataFrame with columns being features
        catCols = list of categorical features, not to be scaled
        scaler = type of sklearn scaler to us
    output:
        Xscaled = scaled DataFrames
    """
    X4keep = df[catCols] if catCols else None
    X4scale = df.drop(X4keep.columns, axis=1, inplace=False) if catCols else df
    if scaler == "minMax":
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(X4scale)
    elif scaler == "stand":
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X4stand)
    elif scaler == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True).fit(X4scale)
    else:
        raise("\nScaler specified unknown!")
    arrXscaled = scaler.transform(X4scale)
    Xscaled = pd.concat(objs=[pd.DataFrame(arrXscaled, index=df.index, columns=X4scale.columns), X4keep], axis=1)
    Xscaled.index.name = "Index"
    return scaler, Xscaled


def getLowVarCols(df, skipCols=None, varThresh=0.0, autoRemove=False):
    """
    Wrapper for sklearn VarianceThreshold.
    input:
        df = DataFrame with columns being features
        skipCols = columns to be skipped
        varThresh = low variance threshold for features to be detected
        autoRemove = Boolean indicator for automatic removal of low variance columns
    output:
        df = DataFrame with low variance features removed
        lowVarCols = list of low variance features
    """
    print("\n  Finding features with low-variance, threshold: {0}".format(varThresh))
    try:
        allCols = df.columns
        if skipCols:
            remainCols = allCols.drop(skipCols)
            maxIdx = len(remainCols) - 1
            skippedIdx = [allCols.get_loc(col) for col in skipCols]

            # adjust insert location by the number of columns removed (for non-zero insertion locations) to keep relative locations intact
            for idx, item in enumerate(skippedIdx):
                if item > maxIdx:
                    diff = item - maxIdx
                    skippedIdx[idx] -= diff
                if item == maxIdx:
                    diff = item - len(skipCols)
                    skippedIdx[idx] -= diff
                if idx == 0:
                    skippedIdx[idx] = item
            skippedVals = df.iloc[:, skippedIdx].values
        else:
            remainCols = allCols

        X = df.loc[:, remainCols].values
        vt = VarianceThreshold(threshold=varThresh)
        vt.fit(X)
        keepColsIdxs = vt.get_support(indices=True)
        keepCols = [remainCols[idx] for idx, _ in enumerate(remainCols) if idx in keepColsIdxs]
        lowVarCols = list(np.setdiff1d(remainCols, keepCols))
        print("    Found {0} low-variance columns.".format(len(lowVarCols)))

        if autoRemove:
            print("    Removing low-variance features...")
            X_removed = vt.transform(X)
            print("    Reassembling the dataframe (with low-variance features removed)...")
            df = pd.DataFrame(data=X_removed, columns=keepCols)
            if skipCols:
                for (i, index) in enumerate(skippedIdx): df.insert(loc=index, column=skipCols[i], value=skippedVals[:, i])
            print("    Succesfully removed low-variance columns: {0}.".format(lowVarCols))
        else:
            print("    No changes have been made to the dataframe.")
    except Exception as e:
        print(e)
        print("    Could not remove low-variance features. Something went wrong.")
    return df, lowVarCols


def getHighCorCols(df, corrThresh=0.95, method="spearman"):
    """
    Compute correlation matrix using pandas
    input: 
        df = input DataFrame with columns being features
        corrThresh = threshold to identify highly correlated features
        method = method to compute DataFrame correlation
    output: 
        corrMat = correlation matrix of all features in input DataFrame
        highCorCols = tuples of highly correlated features
    """ 
    print("\n  Finding features highly-correlated with each other, threshold: {0}".format(corrThresh))
    corrMat = df.corr(method=method, min_periods=1)
    corrMatUpper = corrMat.where(np.triu(np.ones(corrMat.shape), k=1).astype(np.bool))  # Select upper triangle of correlation matrix
    highCorCols = [(row, col) for col in corrMatUpper.columns for row in corrMatUpper.index if corrMatUpper.loc[row, col] > corrThresh]
    print("    Highly correlated columns: {0}".format(highCorCols))
    return corrMat, highCorCols


def plotCorrMat(corrMat, figSize=(8, 8), figName=None):
    """
    Wrapper for sklearn VarianceThreshold.
    input:
        corrMat = correlation matrix of all features in input DataFrame
        figSize = size of figure
        figName = path to save figure
    """
    cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=50, as_cmap=True)
    cg = sns.clustermap(data=corrMat.abs().mul(100).astype(float), cmap='Blues', metric='correlation',  figsize=figSize)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0);
    if figName: plt.savefig(figName, dpi=300, bbox_inches='tight')


def autoDrop(highCorCols, verbose=True):
    """
    Automate the selection of highly-correlated features to be dropped. 
    Rank given to each feature is based on its degree of utility/ease of interpretation if it turns out to be correlated with target labels
    'x', 'y', 'z' not very useful even if found to be important
    'avg' values of bond geometries are more useful than 'num' values
    'max', 'min' of bond geometries are hard to control experimentally
    'angParam', 'centParam' are easier to interpret than 'entroParam'
    'disorder' parameters are easier to interpret than 'q' Steinhardt's parameters
    averaged 'q' parameters are more robust to thermal fluctuations, hence more useful than pure 'q' parameters
    q6 > q4 > q2 == q8 == q10 == q12 in usefulness based on literature, thus 'disorder' parameters follow same sequence
    input:
        highCorCols = list of tuples of highly-correlated features
        verbose = Boolean indicator for output printing
    output: 
        cols2drop = list of features to be dropped
    """
    utilityRanking = {'x': 10, 'y': 10, 'z': 10, 'rad': 0, 
                     'blavg': 2, 'blmax': 8, 'blmin': 8, 'blnum': 3, 
                     'ba1avg': 2, 'ba1max': 8, 'ba1min': 8, 'ba1num': 3,
                     'ba2avg': 2, 'ba2max': 8, 'ba2min': 8, 'ba2num': 3,
                     'btposavg': 2, 'btposmax': 8, 'btposmin': 8, 'btposnum': 3, 
                     'btnegavg': 2, 'btnegmax': 8, 'btnegmin': 8, 'btnegnum': 3, 
                     'cn': 1, 'gcn': 0, 'scn': 3, 'sgcn': 3, 'q6q6': 2, 
                     'Ixx': 5, 'Iyy': 5, 'Izz': 5, 'degenDeg': 6, 
                     'angParam': 4, 'centParam': 4, 'entroParam': 5, 'entroAvgParam': 5.5, 
                     'chi1': 6, 'chi2': 6, 'chi3': 6, 'chi4': 6, 'chi5': 6, 'chi6': 6, 'chi7': 6, 'chi8': 6, 'chi9': 6, 
                     'q2': 5.7, 'q4': 5.6, 'q6': 5.5, 'q8': 5.7, 'q10': 5.7, 'q12': 5.7, 
                     'q2avg': 5.2, 'q4avg': 5.1, 'q6avg': 5, 'q8avg': 5.2, 'q10avg': 5.2, 'q12avg': 5.2, 
                     'disord2': 4.7, 'disord4': 4.6, 'disord6': 4.5, 'disord8': 4.7, 'disord10': 4.7, 'disord12': 4.7, 
                     'disordAvg2': 4.2, 'disordAvg4': 4.1, 'disordAvg6': 4, 'disordAvg8': 4.2, 'disordAvg10': 4.2, 'disordAvg12': 4.2
                     }  # Lower score = Higher rank

    # occurCount = Counter(list(sum(highCorCols, ())))
    print("\n    Sorting all highly-correlated feature pairs based on their minimum and total utility rankings.")
    highCorColsProps = []
    for (col1, col2) in highCorCols:
        rank1, rank2 = utilityRanking[col1], utilityRanking[col2]
        highCorColsProps.append((min(rank1, rank2), rank1 + rank2))
    sortedIdx = sorted(range(len(highCorColsProps)), key=lambda i: highCorColsProps[i])
    highCorCols, highCorColsProps = [highCorCols[i] for i in sortedIdx], [highCorColsProps[i] for i in sortedIdx]
    
    print("\n    Removing one of each highly-correlated feature pairs.")
    cols2drop = []
    for (i, (col1, col2)) in enumerate(highCorCols):
        if verbose: print("      Feature pairs: {0} {1}".format(col1, col2))
        if col1 in cols2drop or col2 in cols2drop:
            if verbose: print("        One of the features is dropped, skip this pair.\n")
            continue
        elif utilityRanking[col1] > utilityRanking[col2]:
            print("        {0} has lower utility score compared to {1}".format(col1, col2))
            cols2drop.append(col1)
        else:
            print("        {0} has lower utility score compared to {1}".format(col2, col1))
            cols2drop.append(col2)
    print("    Feature columns to drop: {0}".format(cols2drop))
    return cols2drop


def varCorrDropCols(X, varThresh=0.01, corrThresh=0.95, verbose=True, figName=None):
    """
    Remove features with low variance and one of the highly-correlated feature pairs
    input:
        X = input scaled DataFrame with each column being feature
        varThresh = threshold below which feature is removed
        corrThresh = threshold above which one of each pair of correlated features is removed
        verbose = Boolean indicator for output printing
    output:
        XNoLVHC = DataFrame with the undesired features removed
        corrMatNoLVHC = computed correlated matrix
    """
    # Remove columns with low variance
    XNoLV, lowVarCols = getLowVarCols(df=X, skipCols=None, varThresh=varThresh, autoRemove=True)  # Using min-max scaled for now
    
    # Remove one of the feature columns that are highly correlated with each other
    corrMatNoLV, highCorCols1 = getHighCorCols(df=XNoLV, corrThresh=corrThresh, method=corrMethod)
    cols2drop = autoDrop(highCorCols1, verbose=verbose)
    XNoLVHC = XNoLV.drop(labels=cols2drop, axis=1, index=None, columns=None, level=None, inplace=False, errors="raise")
    corrMatNoLVHC, highCorCols2 = getHighCorCols(df=XNoLVHC, corrThresh=corrThresh, method=corrMethod)
    plotCorrMat(corrMat=corrMatNoLVHC, figSize=(8, 8), figName=figName)
    return XNoLVHC, corrMatNoLVHC


def plot_rho_delta(density, delta_matrix):
    '''
    Draw the decision graph
    INPUTS:
        density: density of N points
        delta_matrix: computing the minimum distance between the point and any other with high density
    OUTPUTS:
        density and delta plot
    '''
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121)
    for i in range(len(density)):
        plt.scatter(x=density[i], y=delta_matrix[i], c='k', marker='o', s=15)
    plt.xlabel('density')
    plt.ylabel('delta')
    plt.title('Decision graph')
    plt.sca(ax1)
    plt.show()
    

def combine_clustering(data, gamma, k=3):
    gamma_idxs = gamma.argsort() 
    gamma_idxs = gamma_idxs[::-1]
    
    # TODO: Choose optimal k automatically *******  Would unnormalised gamma be useful? 
    # gammaDiff = np.subtract(gamma[idx[:-1]], gamma[idx[1:]])
    
    idxs = [data.index[idx] for idx in gamma_idxs[:k]]
    newL, orderedL, count = cfs.applyILS(data, idxs)
    return newL, orderedL, count, idxs


def test_cfsdp(data, X_embedded, k=3):
    density, neighDensPoints, scores = pre_calculate(data)
    # Plot rho vs delta, t-SNE distribution, gamma graph
    cfs.density_delta(density, neighDensPoints, X_embedded)
    cfs.plot_center(neighDensPoints, density, scores)  
    
    newL, orderedL, count, idxs = combine_clustering(data, scores, k=k)
    cfs.draw_ILS(count, X_embedded, newL, ut.colors)
    cfs.plot_centroid(X_embedded, idxs)
    return newL, orderedL, count, idxs, scores


def locateMinBtwPeaks(orderedL, peaksRanges):
    """
    Identify the minima in each specified cluster (indicated by ranges to find peaks) -- Not used when 
    automatic peak finding algorithm is used.
    
    Parameters:
        orderedL : output DataFrame from ILS plot
        peaksRanges : list of tuples containing start and end point of the range where the peak(s) lie
    Outputs:
        endPoints : list of end points of each cluster, correspond to ILS x-axis
        minIdxs : list of cluster minima, ID correspond to ILS x-axis
        ILSclust : Series of ILS cluster labels
        ILSorder : output DataFrame from ILS containing labelling order of ILS clusters
    """
    startPoint, endPoints, minIdxs = 0, [], []
    numClusters = len(peaksRanges) + 1
    for i in range(numClusters):
        # Identify peak(s) as clusters separation point(s)
        if i != numClusters - 1:
            # print(orderedL.iloc[peaksRanges[i][0]:350,:]["minR"].idxmax())
            endIdx = orderedL.iloc[peaksRanges[i][0]:peaksRanges[i][1], :]["minR"].idxmax()
            endPoint = int(orderedL.loc[endIdx, :]['order'])
        else:
            endPoint = len(orderedL)
        minIdx = orderedL.iloc[startPoint:endPoint + 1, :]["minR"].idxmin()
        minPoint = int(orderedL.loc[minIdx, :]['order'])
        endPoints.append(endPoint)
        minIdxs.append(minIdx)
        startPoint = endPoint + 1
        print("Cluster {0} end point: {1}\nMinima within cluster: {2} (ID: {3})".format(i, endPoint, minPoint, minIdx))
    return endPoints, minIdxs


def findPeaksV1(minR, winSize, sigThreshCoef=0.1, visThresh=0.1):
    """
    Locate maxima in a series of values.
    
    Parameters:
        minR : series of minimum distance values as obtained from ILS clustering output
        winSize : integer indicating range of values to be averaged over (need to be divisible by 2)
        sigThreshCoef : float indicating threshold for significance, to be multiplied to the noise range in a given window
        visThresh : float indicating threshold for visibility
    Outputs: 
        maxima : numpy array containing maxima in the series (in terms of order instead of index)
    """
    maxima = []
    rangeR = (minR.max()-minR.min())
    for (i, R) in enumerate(minR):
        if (i == 0) or (i == len(minR) - 1): continue
        start = 0 if i-winSize/2 < 0 else int(i-winSize/2)
        end = None if i+winSize/2 > len(minR) else int(i+winSize/2)
        window = minR[start:end]
        if R != window.max(): continue  # Only maxima can pass
        if (window == R).sum() != 1: continue  # Only unique maximum within window can pass
        
        # Locate the peak within the window
        if i-winSize/2 < 0: peakPos = 0 - (i-winSize/2)
        elif i+winSize/2 > len(minR): peakPos = (i+winSize/2) - len(minR)
        else: peakPos = winSize / 2
        
        # windowNoPeak = np.delete(np.array(window[:int(peakPos) + 1]), int(peakPos), axis=None)  # Remove the peak from the window
        windowNoPeak = np.array(window[:int(peakPos)])
        sigThresh = (windowNoPeak.max()-windowNoPeak.min()) * sigThreshCoef + windowNoPeak.max()
        if R < sigThresh: continue  # Only peaks that're significantly distinguished from previous points can pass
        visibility = window.max() - window.min() / rangeR
        if visibility < visThresh: continue  # Only peaks that're typically visible can pass
        maxima.append(i)
    return np.array(maxima)


def findPeaksV2(minR, halfWinSize, peakFunc='S2', sigConst=1.2, verbose=False):
    """
    Locate maxima in a series of values.
    
    Parameters:
        minR : series of minimum distance values as obtained from ILS clustering output (np array of length N)
        halfWinSize (k) : integer indicating half of the range of values to be averaged over
        sigConst (h): significance constant, typically 1<=h<=3
    Outputs: 
        maxima : numpy array consisting of detected peaks (in terms of order instead of index)
    """
    if verbose: print("Using peak function {0}".format(peakFunc))
    
    peakFuncVals = np.zeros(len(minR))
    for (i, R) in enumerate(minR):
        if i == 0 or i == len(minR)-1: continue
        
        # Extract neighbouring points from both sides
        leftStart = 0 if i-halfWinSize < 0 else i-halfWinSize
        rightEnd = None if i+halfWinSize > len(minR) else i+halfWinSize
        leftWindow, rightWindow = minR[leftStart:i], minR[i+1:rightEnd]
        leftDiff, rightDiff = R - leftWindow, R - rightWindow 
        
        # Compute peak functions (check if float divisions are fine!)
        if peakFunc == 'S1':
            peakFuncVal = (leftDiff.max()+rightDiff.max()) / 2
        elif peakFunc == 'S2':
            peakFuncVal = (leftDiff.sum()/halfWinSize+rightDiff.sum()/halfWinSize) / 2
        elif peakFunc == 'S3':
            peakFuncVal = (R-leftWindow.sum()/halfWinSize+R-rightWindow.sum()/halfWinSize) / 2
        else:
            raise AssertionError("Peak function specified is unknown!")
        
        peakFuncVals[i] = peakFuncVal
    
    # Peak candidates have positive values of peak functions
    posPeakFuncVals = peakFuncVals[peakFuncVals > 0]
    posPeakFuncMean, posPeakFuncStd = posPeakFuncVals.mean(), posPeakFuncVals.std()
    # print(posPeakFuncVals, posPeakFuncMean, posPeakFuncStd)
    
    maxima, prevPeakPos = [], -halfWinSize
    for (peakPos, peakFuncVal) in enumerate(peakFuncVals):
        if peakFuncVal > 0 and peakFuncVal-posPeakFuncMean > sigConst*posPeakFuncStd:
            if verbose: print("  Found qualified peak candidate:", peakPos)
            # Choose more significant peak if peak found within window range of previous peak
            if (peakPos < prevPeakPos + halfWinSize) and (len(maxima) > 0):
                if verbose: print("    But too close to previous peak, comparing both peaks...")
                if peakFuncVal > peakFuncVals[maxima[-1]]: 
                    if verbose: print("    Previous peak is removed.")  # peakFuncVal, peakFuncVals[maxima[-1]]
                    maxima.pop()
                else: 
                    if verbose: print("    Retaining previous peak.")
                    continue
            maxima.append(peakPos)
            prevPeakPos = peakPos
    # Optional condition to remove last peak if the cluster is too small
    # if maxima[-1] + halfWinSize > len(minR): maxima.pop()  # Doesn't need to be 'halfWinSize' here
        
    return np.array(maxima)


def locateMinAutoV1(minR, winSize):
    """
    Locate minima in a series of values (hopefully densest point) -- could potentially use gamma graph to locate.
    
    Parameters:
        minR : series of minimum distance values as obtained from ILS clustering output
        winSize : integer indicating range of values to be averaged over (need to be divisible by 2)
    Outputs:
        endPoints : list of maxima in the series + the last point of the series (in terms of order instead of index)
        minima : list of minima in the series (in terms of order instead of index)
    """
    index = np.arange(len(minR))
    maxima = findPeaks(minR=minR, winSize=winSize)
    # filtered = gaussian_filter1d(minR, winSize)  # Smoothen the series of values
    # maxima = find_peaks_cwt(filtered, len(filtered) * [winSize])

    # removeIdxs = []
    # for (i, maximum) in enumerate(maxima):
    #     if i == 0:
    #         if maximum - 0 < winSize: removeIdxs.append(i)
    #     elif i == len(maxima) - 1:
    #         if len(minR) - maximum < winSize: removeIdxs.append(i)
    #     else:
    #         if maximum - maxima[i-1] < winSize: removeIdxs.append(i)
    # maxima = np.delete(maxima, removeIdxs)
    
    betweenMax, betweenIndex = np.split(minR, maxima), np.split(index, maxima)
    # betweenMax, betweenIndex = np.split(filtered, maxima), np.split(index, maxima)
    subMinVals, subMinIdxs = [min(i) for i in betweenMax], [np.argmin(i) for i in betweenMax]
    minima = [betweenIndex[i][subMinIdxs[i]] for i in range(len(subMinIdxs))]
    endPoints = [betweenIndex[i][-1] for i in range(len(subMinIdxs))]
    print("End points: {0}\nMinima: {1}".format(endPoints, minima))
    return endPoints, minima


if __name__ == "__main__":
    # Attempt to plot gamma graph on log scale for better decision of k (number of clusters)
    diff = np.subtract(scores[idx[::-1][:-1]], scores[idx[::-1][1:]])
    plt.scatter(range(len(diff)), diff, c='k', marker='o', s=10) #-np.sort(-diff) * 100)
    ax = plt.gca()
    ax.set_yscale('log')

    # Snippet to test functions related to CFSDP
    dfIdx = 0  # 0 to 38, different nanoparticles (datasets)
    combIdx = 0  # 0 to 30, different feature sets
    combinations = combList[combIdx] + ["surf"]
    featCols = [feat for feat in combinations if feat in dfScaledNoLVHCs[dfIdx].columns]
    df = dfScaledNoLVHCs[dfIdx][featCols]
    X_embedded = df[df["surf"] == 1]
    X_embedded = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1).fit_transform(X_embedded)
    X_embedded = pd.DataFrame(X_embedded, columns = ['x', 'y'])
    newL, count, index = test_cfsdp(df[df["surf"] == 1], X_embedded)
    
    # Potentially Useful tsfresh features
    # absolute_maximum()
    # count_above()
    # count_above_mean()
    # fft_aggregated()
    # first_location_of_maximum()
    # last_location_of_maximum()
    # has_duplicate_max()
    # large_standard_deviation()
    # maximum()
    # number_cwt_peaks()
    # number_peaks()
    # symmetry_looking()
    # value_count()