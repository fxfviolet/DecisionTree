from math import *
import operator
import matplotlib.pyplot as plt
import pylab
pylab.mpl.rcParams['font.sans-serif']=['SimHei']

# 加载数据
def load_data(filename):
    file = open(filename)
    featureMat = file.readline().strip().split(',')  # 标题作为特征(属性)
    dataMat = []
    k = 1
    for line in file.readlines():
        line = line.strip().split(',')
        data = [k]                                 # 对每一行数据添加一个编号
        for i in range(1,len(line)):
            data.append(line[i])
        dataMat.append(data)
        k += 1
    return dataMat,featureMat

# 计算香农熵
def calcShannonEnt(dataMat):
    labelCount = {}
    for d in dataMat:
        label = d[-1]                             # 最后一列是类别
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    shannonEnt = 0.0
    for key in labelCount.keys():
        prob = float(labelCount[key])/len(dataMat) # 每一种类别的比例
        shannonEnt -= prob * log(prob,2)           # 香农熵
    return shannonEnt

# 按照给定特征划分数据集
def splitData(dataMat,axis,item):
    dataRet = []
    for d in dataMat:
        if d[axis] == item:           # 找出特征是'item'的列
            data = d[:axis]
            data.extend(d[axis+1:])
            dataRet.append(data)
    return dataRet 

# 按照给定特征生成新的特征
def splitFeature(featureMat,axis,feature):
    if featureMat[axis] == feature:        # 找出特征是'feature'的特征
        newFeature = featureMat[:axis]
        newFeature.extend(featureMat[axis+1:])
    return newFeature

# 找出信息增益最大的特征
def chooseBestFeatureToSplit(dataMat):
    baseEnt = calcShannonEnt(dataMat)      # 经验熵
    bestGain = 0
    bestFeature = 0
    for i in range(1,len(dataMat[0])-1):  # 第1列是编号，最后一列是类别，只对数据集进行计算
        data = [d[i] for d in dataMat]    # 某一列的所有特征属性
        data = set(data)                   # 该列属性的集合，即去除重复的属性
        newEnt = 0
        for item in data:
            newData = splitData(dataMat,i,item)
            prob = len(newData)/len(dataMat)
            newEnt += prob * calcShannonEnt(newData)      # 经验条件熵
        infoGain = baseEnt - newEnt                       # 信息增益
        if infoGain > bestGain:         # 找出最大的信息增益
            bestGain = infoGain
            bestFeature = i             # 信息增益最大的列
    return bestFeature

# 找出出现次数最多的类别
def majorityCnt(featureList):
    featureCount = {}
    for vote in featureList:
        if vote not in featureCount.keys():
            featureCount[vote] = 0
            featureCount[vote] += 1
    sortedClassCount = sorted(featureCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 递归构建决策树
def createTree(dataMat,featureMat):
    featureList = [d[-1] for d in dataMat]                     # 最后一列即为所有类别
    if featureList.count(featureList[0]) == len(featureList):   # 如果只有一种类别，不再划分，返回该类别
        return featureList[0]
    if len(dataMat[0]) == 1:                                # 遍历完所有特征后返回出现次数多的类别
        return majorityCnt(featureList)
    bestFeature = chooseBestFeatureToSplit(dataMat)         # 找出信息增益最大的特征
    bestFeatureName = featureMat[bestFeature]               # 信息增益最大的特征名称
    myTree = {bestFeatureName:{}}                           # 构建树字典
    data = [d[bestFeature] for d in dataMat]
    data = set(data)
    for item in data:
        newData = splitData(dataMat,bestFeature,item)
        newLabel = splitFeature(featureMat,bestFeature,bestFeatureName)
        myTree[bestFeatureName][item] = createTree(newData,newLabel)     # 递归构建决策树
    return myTree 

# 绘制树图形
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:  
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

dataMat,featureMat = load_data('test_data.txt')
myTree = createTree(dataMat,featureMat)
createPlot(myTree)   

        
        