# -*- coding: utf-8 -*-

# **********************************************************************************************************************
# This file is part of the Spatial point Pattern Analysis Algorithm, and is used for  spatial point clustering analysis.
# And it could be used as assist tool to planning decision of decentralized sewage treatment facilities. This model
# contains mainly three parts, they are points trend analysis, point cluster analysis and spatial visualization.
#
# Author: Yuansheng Huang
# Date: 2019.09.24
# Version: V 0.1
# Note: In V 0.1 we use 2d Euclidean distance to represent the distance between two points.

# Reference: Clark and Evans, 1954; Gao, 2013

# **********************************************************************************************************************

# General import
import arcpy
import copy
import math
import numpy as np
from scipy.spatial import Delaunay


# ----------------------------------------------------------------------------------------------------------------------
# 以下函数用于空间点的趋势分析，采用最邻近指数法
# ----------------------------------------------------------------------------------------------------------------------

def readArea(areaShape):
    """
    读取shapefile中研究区域的面积

    输入参数
    areaShape: 研究区域矢量地图，用于读取面积值。

    输出参数
    area: 研究区域面积
    """
    areaList = []
    rows = arcpy.SearchCursor(areaShape)
    fields = arcpy.ListFields(areaShape)
    for row in rows:
        for field in fields:
            if field.name == "area":
                A = row.getValue(field.name)
        areaList.append(A)
    Area = np.sum(areaList)
    return Area


def readSpatialPoints(pointShape):
    """
    读取空间点坐标数据，并保存为列表

    输入参数
    in_FC: Path to point shapefile

    输出参数
    pointList: 空间点坐标列表，[[X,Y],...]
    pX, pY: X, Y值列表
    spatialRef: 空间参考
    """
    pointList, rows, fields = [], arcpy.SearchCursor(pointShape), arcpy.ListFields(pointShape)
    spatialRef = arcpy.Describe(pointShape).spatialReference
    for row in rows:  # Iterate shapefile
        for field in fields:
            if field.name == "POINT_X":
                X = row.getValue(field.name)
            if field.name == "POINT_Y":
                Y = row.getValue(field.name)
        pointList.append([X, Y])

    if len(pointList) < 1:
        raise Exception("EMPTY LIST: YOU GOT AN EMPTY LIST, PLEASE CHECK YOUR INPUT FILE!!!")
    return pointList, spatialRef


def nearestDistance(pointList):
    """
    此函数用于计算各点到其最近点间的距离，并返回距离列表

    输入参数
    pointList: 空间点坐标列表，[[X,Y],...]

    输出参数
    distanceList: 最近距离列表[l1,l2,l3,l4,...]
    """
    distanceList = []
    for i in range(len(pointList)):
        pointLength = []
        for j in range(len(pointList)):
            if i != j:
                length = math.hypot(pointList[i][0] - pointList[j][0], pointList[i][1] - pointList[j][1])
                pointLength.append(length)
            else:
                continue
        distance = min(pointLength)
        distanceList.append(distance)

        if len(distanceList) < 1:
            raise Exception("EMPTY LIST: YOU GOT AN EMPTY LIST, PLEASE CHECK YOUR INPUT FILE!!!")
    return distanceList


def NNI(pointList, distanceList, area):
    """
    用于计算空间点集的最邻近指数。当NNI>1时，空间点集呈均匀分布，当NNI<1时，空间点集呈聚集分布

    输入参数
    pointList: 空间点坐标列表，[[X,Y],...]

    输出参数
    index:  空间点集的最邻近指数
    z_test: z检验数值
    """
    N = len(pointList)
    ran = 0.5 * math.sqrt(area / N)

    for e in range(len(distanceList)):  # 替换显著差异值
        if distanceList[e] > np.mean(distanceList):
            distanceList[e] = ran
        else:
            continue

    sumD = np.sum(distanceList)
    SE = 0.26236 / (math.sqrt(N ** 2) / area)
    index = (sumD / N) / ran
    z_test = ((sumD / N) - ran) / SE
    return index, z_test


# ----------------------------------------------------------------------------------------------------------------------
# 以下函数基于Delaunay Triangle对空间点进行初步聚类，即删除DT的全局长边
# ----------------------------------------------------------------------------------------------------------------------

def getClusterID(pointList):
    """
    获取簇ID号列表，返回一级聚类簇ID和二级聚类簇ID， 用于后续聚类簇的编号
    """
    clusterID1 = ["A" + str(i) for i in range(len(pointList))]
    clusterID2 = ["B" + str(i) for i in range(len(pointList))]
    return clusterID1, clusterID2

def deleteElements(liste):  # todo 删除此函数，用 list(set())代替
    """
    删除列表中重复的元素，返回无重复元素的列表
    """
    listX = []
    for i in liste:
        if i not in listX:
            listX.append(i)
    return listX


def delaunayTrangle(pointList):
    """
    获取空间点集points的Delaunay Trangle (DT) 及DT的顶点索引和坐标。

    输入参数
    pointList: 空间点坐标列表[[X, Y], ...]

    输出参数
    indexList: DT顶点索引列表，如[[1,2,3],...]
    coordinateList: DT顶点坐标列表[[[x1,y1],[x2,y2],[x3,y3]],...]
    DT: Delaunay Trangle，由SciPy spatial 中的Delaunay得到
    vertexPoints: points coordinate with ID. [[id1,[x1,y1]],...]，不含重复点   pointList基础上增加索引ID
    """
    points = np.array(pointList)
    DT = Delaunay(points)
    indexList = DT.simplices[:].tolist()
    coordinateList, vertexPointsAll, vertexPoints = [], [], []
    for i in range(len(indexList)):
        b = points[DT.simplices[i]]
        a = b.tolist()
        coordinateList.append(a)
        for j in range(len(a)):
            vertexPointsAll.append([indexList[i][j], a[j]])
    vertexPoints = deleteElements(vertexPointsAll)
#    for k in vertexPointsAll:  # TODO 需修改核实
#        if k not in vertexPoints:
#            vertexPoints.append(k)

    return indexList, coordinateList, DT, vertexPoints


def getID(a, b):
    """
    根据DT顶点索引生成DT边ID号，只获取一条边的ID号，而非ID列表。

    输入参数
    a, b: DT顶点索引号

    输出参数
    edgeID: DT边索引号
    """
    if a == b:
        raise Exception("ERROR: Indexes point to the same point!!!")
    maxIndex, minIndex = max(a, b), min(a, b)
    edgeID = "V" + "_" + str(minIndex) + "_" + str(maxIndex)
    return edgeID


def getIndex(edgeID):
    """
    从edgeID中获取各顶点的索引号
    """
    liste = edgeID.split("_")
    indexListStr = liste[1:]
    indexList = [int(i) for i in indexListStr]
    indexA, indexB = indexList[0], indexList[1]
    return indexA, indexB


def getTriangleEdge(indexList, coordinateList):
    """
    获取DT的欧氏距离边长列表，用于计算整体边长均值和整体边长变异。

    输入参数
    indexList: DT顶点索引列表，如[[1,2,3],...]
    coordinateList: DT顶点坐标列表[[[x1,y1],[x2,y2],[x3,y3]],...]

    输出参数
    edgeListTriangle: 所有DT边长列表（每个三角形一个元素）[[[1, 1A, 1B, len1], [2, 2A, 2B, len2], [3, 3A, 3B, len3]],....]
    edgeListAll: 所有DT边长列表（含重复边，按边列出）[[id, ida,idb,len],...]
    edgeList: 去除重复的DT边列表（不含重复边）[[id, ida,idb,len],...] id为边编号，ida/idb为边顶点索引号
    """
    edgeListTriangle, edgeListAll, edgeList = [], [], []
    for i in range(len(indexList)):  # 获取edgeListTriangle列表
        indexI = indexList[i]
        indexC = coordinateList[i]
        a, b, c = indexI[0], indexI[1], indexI[2]  # index
        m, l, n = indexC[0], indexC[1], indexC[2]  # coordinate
        ID1, ID2, ID3 = getID(a, b), getID(a, c), getID(b, c)  # 获取边的ID号
        ID1A, ID1B = getIndex(ID1)  # 获取边顶点索引号，与vertexPoints中ID对应
        ID2A, ID2B = getIndex(ID2)
        ID3A, ID3B = getIndex(ID3)
        len1 = math.hypot(m[0] - l[0], m[1] - l[1])  # 获取边长
        len2 = math.hypot(m[0] - n[0], m[1] - n[1])
        len3 = math.hypot(l[0] - n[0], l[1] - n[1])
        edgeListTriangle.append([[ID1, ID1A, ID1B, len1], [ID2, ID2A, ID2B, len2], [ID3, ID3A, ID3B, len3]])

    for j in range(len(edgeListTriangle)):  # 获取edgeListAll列表
        edge = edgeListTriangle[j]
        for k in range(len(edge)):
            edgeListAll.append(edge[k])
    edgeList = deleteElements(edgeListAll)  # 获取edgeList列表
#    for s in edgeListAll:
#        if s not in edgeList:
#            edgeList.append(s)
#        else:
#            continue
    return edgeListTriangle, edgeListAll, edgeList


def getNeighbourhood1(vertexPoints, edgeList):
    """
    用于获取vertex的一阶邻域顶点，并返回各点的一阶邻域边长均值（用于计算二阶邻域边长均值）

    输入参数
    vertexPoints: points coordinate with ID. [[id1,[x1,y1]],...]，不含重复点
    edgeList: 去除重复的DT边列表（不含重复边）[[id, ida,idb,len],...] id为边编号，ida/idb为边顶点索引号, 由getTriangleEdge函数计算

    输出参数
    neighbourhood1AVG: 一阶邻域边长均值[[pointID, mean],...]
    firstOrderPoint: 一阶邻域顶点索引列表[[p1,p2,...pointID,....pn, pointID],...]
    """
    neighbourhood1AVG = []  # 各点一阶邻域边长均值列表
    firstOrderPoint = []  # i点一阶邻域点列表 [a,b,c,d....]
    for i in vertexPoints:  # iteration
        length1 = []  # 一阶邻域边长
        firstPointX = []
        for j in edgeList:  # iteration
            if i[0] in j[1:3]:  # 一阶邻域条件：点P的索引i[0]包含在与其相邻边的顶点索引中
                length1.append(j[-1])
                firstPointX.append(j[1])  # 用于计算二阶邻域边长均值
                firstPointX.append(j[2])
            else:
                continue
            # for X in firstPointX:  # 获取i点及其一阶邻域点的索引， 含重复点
                # for x in X:
                # if x not in point:
            # point.append(x)
        firstPoint = deleteElements(firstPointX)  # 获取不含重复索引的一阶领域索引列表

        firstPoint.append(i[0])  # 将i点的索引添加至嵌套字列表的末尾
        neighbourhood1AVG.append([i[0], np.mean(length1)])
        firstOrderPoint.append(firstPoint)
    return neighbourhood1AVG, firstOrderPoint


def getNeighbourhood2(vertexPoints, edgeList, firstOrderPoint):
    """
    获取vertexPoints列表各点的二阶邻域边长均值，任意点P的一阶邻域点的各条边包含点P二阶邻域边和点P一阶邻域三角形的所有边。所以计算二阶
    邻域时，先求点P所有一阶邻域点的所有边，再取相互不重复的边即可。

    输入参数
    vertexPoints: points coordinate with ID. [[id1,[x1,y1]],...]，不含重复点
    edgeList: 去除重复的DT边列表（不含重复边）[[id, ida,idb,len],...] id为边编号，ida/idb为边顶点索引号
    firstOrderPoint: 一阶邻域顶点列表[[p1,p2,...pointID,....pn, pointID],...]

    输出参数
    neighbourhood2AVG: 二阶邻域边长均值[[pointID, mean],...]
    """
    neighbourhood2AVG = []
    for i in vertexPoints:  # 迭代顶点
        secondPoint = []
        edge = []
        length2 = []
        for j in firstOrderPoint:  # 获取i点的一阶邻域点
            if i[0] == j[-1]:
                secondPoint = j[:-1]
        for a in secondPoint:  # 获取i点所有一阶邻域点的一阶邻域边
            for b in edgeList:
                if a in b:
                    edge.append([a, b])  # [[a,[id,a,b,len]],...]
        for l in edge:
            for n in edge:
                if l[0] != n[0] and not l[0] in n[1]:
                    length2.append(l[1][-1])
        neighbourhood2AVG.append([i[0], np.mean(length2)])
    return neighbourhood2AVG


def calcGlobalStatistic(edgeList):
    """
    获取DT整体边长均值、整体边长变异（STD）、任一顶点的一阶邻域边长均值
    """
    edge = []
    for i in edgeList:  # 获取边长数值
        edge.append(i[-1])
    edgeAVG = np.mean(edge)
    edgeSTD = np.std(edge)
    return edgeAVG, edgeSTD


def globalCut(vertexPoints, edgeList):
    """
    获取各顶点的全局约束值

    输入参数
    vertexPoints: points coordinate with ID. [[1,[x1,y1]],...]，不含重复点
    edgeList: 去除重复的DT边列表（不含重复边）[[id, ida,idb,len],...] id为边编号，ida/idb为边顶点索引号, 由getTriangleEdge函数计算

    输出参数
    globalCutList: 全局约束列表[pointIndex, cutValue],...]
    """
    globalAVG, globalSTD = calcGlobalStatistic(edgeList)
    firstOrderEdgeAVG, _ = getNeighbourhood1(vertexPoints, edgeList)  # todo
    globalCutList = []
    for i in vertexPoints:
        mean1 = 0
        for j in firstOrderEdgeAVG:
            if i[0] == j[0]:  # pointIndex:i[0], j[0]
                mean1 = j[1]
        cutValue = globalAVG + 1 * (globalAVG / mean1) * globalSTD
        globalCutList.append([i[0], cutValue])
    return globalCutList


def deleteLongEdge(edgeList, globalCutList):  # TODO
    """
    删除全局长边，并返回全局长边和全局其他边列表

    输入参数
    edgeList: 去除重复的DT边列表（不含重复边）[[id, ida,idb,len],...] id为边编号，ida/idb为边顶点索引号, 由getTriangleEdge函数计算
    globalCutList: 全局约束列表[pointIndex, cutValue],...]

    输出参数
    otherEdgeList，longEdgeList: 全局短边，全局长边
    """
    otherEdgeListX, longEdgeListX = [], []
    for i in globalCutList:
        for j in edgeList:  # 获取各顶点的other edges
            if i[0] in j and i[1] > j[-1]:  # todo
                otherEdgeListX.append(j)
            if i[0] in j and i[1] <= j[-1]:
                longEdgeListX.append(j)

    otherEdgeList = deleteElements(otherEdgeListX)  # 去除列表中重复元素
    longEdgeList = deleteElements(longEdgeListX)
#    otherEdgeList = deleteElements(otherEdgeListX)  # 去除列表中重复元素
#    longEdgeList = deleteElements(longEdgeListX)
    return otherEdgeList, longEdgeList


def getIsolatedPoints(vertexPoints, otherEdgeList, mark):
    """
    用于获取空间点集中的孤立点，并给每个点增加标记，在cluster函数中调用。

    输入参数
    vertexPoints: points coordinate with ID. [[1,[x1,y1]],...]，不含重复点
    otherEdgeList: 删除全局长边后的Delaunay三角网格边长[[id, ida,idb,len],...] id为边编号，ida/idb为边顶点索引号
    mark: 用于区分点簇的标记

    输出参数
    otherPointsList: 非孤立点
    isolatedPoints: 孤立空间点列表。[[1,[x1,y1]，mark], []...]
    """
    markerA = mark + str(0)
    other, isolate = [], []
    for vp in vertexPoints:
        pntV = vp
        for e in otherEdgeList:
            if pntV[0] == e[1] or pntV == e[2]:
                other.append(pntV)
            elif pntV[0] != e[1] and pntV != e[2]:
                isolate.append(pntV)
        otherPointsList = [i for i in other if other.count(i) == 1]
        isolatePointList = [j for j in isolate if isolate.count(j) == 1]

    for i in isolatePointList:
        i.append(markerA)

    return isolatePointList, otherPointsList


def aggregation(pointsList):
    """
    用于获取孤立点以外的其他点所构成的点簇，每个点簇所包含的点为一个元素。在cluster函数中调用。
    此函数将嵌套列表中有相同元素的子列表合并，并将索引号较小的一个元素设置为两个子元素的并，较大一个设置为空列表[]。

    输入参数
    pointsList: 嵌套列表

    输出参数
    mergedPoint: 合并后的列表
    """
    mergedPoint = []
    for a in range(len(pointsList)):
        for b in range(len(pointsList)):
            merge = pointsList[a] + pointsList[b]
            X = [i for i in merge if merge.count(i) == 1]
            Y = len(pointsList[a]) + len(pointsList[b])
            if len(X) < Y:
                pointsList[a] = X
                pointsList[b] = []
            elif a == b or pointsList[a] == [] or pointsList[b] == []:
                continue
            elif len(X) == Y and a != b:
                continue

    for i in pointsList:
        if i:
            mergedPoint.append(i)
        else:
            continue
    return mergedPoint


def cluster(vertexPoints, otherEdgeList, mark):  # todo 可以考虑增加一个标记参数，在调用函数是特定输入  黄源生 20191007
    """
    根据全局其他边，初步聚类，给个点添加簇标号，并以列表的形式返回（嵌套列表，在之前vertexPoint的基础上每个元素的末尾添加标识。

    输入参数
    vertexPoints: points coordinate with ID. [[1,[x1,y1]],...]，不含重复点
    otherEdgeList: 删除全局长边后的Delaunay三角网格边长[[id, ida,idb,len],...] id为边编号，ida/idb为边顶点索引号
    mark: 用于区分点簇的标记

    输出参数
    clusterPoints: 删除全局长边后点簇，各点有簇标记，每个元素为一个点簇
    markedPoints: 数据同clusterPoints，数据结构变动，
    """
    # 获取孤立点
    isolatePointList, otherPointsList = getIsolatedPoints(vertexPoints, otherEdgeList, mark)

    # test
    length2 = len(isolatePointList)
    num2 = str(length2)
    E2 = str(isolatePointList)
    arcpy.AddMessage("isolatePointList number: " + num2 + E2)

    length2 = len(otherPointsList)
    num2 = str(length2)
    E2 = str(otherPointsList)
    arcpy.AddMessage("otherPointsList number: " + num2 + E2)

    _, firstOrder = getNeighbourhood1(otherPointsList, otherEdgeList)  # 索引列表
    firstOrderPoints = [i[:-1] for i in firstOrder]  # 获取其他点（不包含孤立点）的一阶邻域点索引

    mergedPoints = aggregation(firstOrderPoints)

    operateVertexPoints = copy.deepcopy(vertexPoints)  # 深度负值，防止反复引用导致数据结构变化
    clusterPointsX, markedPointsX = [isolatePointList], copy.deepcopy(isolatePointList)
    for i in mergedPoints:
        markerB = mark + str(len(i))
        clusterX = []
        for I in i:
            for j in operateVertexPoints:
                if I == j[0]:
                    i.append(markerB)
                    markedPointsX.append(i)
                    clusterX.append(i)
        clusterPointsX.append(clusterX)

    markedPoints = [i for i in markedPointsX if markedPointsX.count(i) == 1]
    clusterPoints = [i for i in clusterPointsX if clusterPointsX.count(i) == 1]
    return clusterPoints, markedPoints


def cluster1(vertexPoints, otherEdgeList, mark):  # todo 可以考虑增加一个标记参数，在调用函数是特定输入  黄源生 20191007
    """
    根据全局其他边，初步聚类，给个点添加簇标号，并以列表的形式返回（嵌套列表，在之前vertexPoint的基础上每个元素的末尾添加标识。

    输入参数
    vertexPoints: points coordinate with ID. [[1,[x1,y1]],...]，不含重复点
    otherEdgeList: 删除全局长边后的Delaunay三角网格边长[[id, ida,idb,len],...] id为边编号，ida/idb为边顶点索引号
    mark: 用于区分点簇的标记

    输出参数
    clusterPoints: 删除全局长边后点簇，各点有簇标记，每个元素为一个点簇
    markedPoints: 数据同clusterPoints，数据结构变动，
    """
    marker1 = mark + str(0)  # 孤立点标记为X0
    isolate, otherPoint, pntX, testPoint = [], [], [], copy.deepcopy( vertexPoints)
    pnt1, pnt2 = [], []
    for pt in testPoint:
        for l in otherEdgeList:
            if pt[0] not in l[1:3]:  # 获取孤立点, 如点不存在于任何边中，则为孤立点，标为"A0"
                pnt1.append(pt)
            else:
                pnt2.append(pt)
        isolate = deleteElements(pnt1)  # 所有孤立点，不含重复。 TODO
        otherPoint = deleteElements(pnt2)

    # test
    length2 = len(isolate)
    num2 = str(length2)
    E2 = str(isolate)
    arcpy.AddMessage("isolate number: " + num2 + E2)

    length2 = len(otherPoint)
    num2 = str(length2)
    E2 = str(otherPoint)
    arcpy.AddMessage("otherPoint number: " + num2 + E2)

    _, firstOrder = getNeighbourhood1(otherPoint, otherEdgeList)  # 索引列表
    firstOrderPoints = []
    for i in firstOrder:  # 获取其他点（不包含孤立点）的一阶邻域点索引,
        otherPoints = i[:len(i) - 1]
        firstOrderPoints.append(otherPoints)

    for a in range(len(firstOrderPoints)):  # 获取otherPoint的一阶邻域点索引
        for b in range(len(firstOrderPoints)):
            X = deleteElements(firstOrderPoints[a] + firstOrderPoints[b])
            Y = len(firstOrderPoints[a]) + len(firstOrderPoints[b])
            if len(X) < Y:
                firstOrderPoints[a] = X
                firstOrderPoints[b] = [0]
            elif a == b or firstOrderPoints[a] == [0] or firstOrderPoints == [0]:
                continue

    clusterPointsX, markedPointsX = [isolate], isolate
    for PI in firstOrderPoints:  # PI: pointIndex, PC: pointCoordinate
        clusterX, length = [], len(PI)
        marker2 = mark + str(length)
        if PI == [0]:
            continue
        elif PI != [0]:
            for E in PI:  # get point index
                for PC in vertexPoints:
                    point = PC
                    if E == point[0]:
                        point.append(marker2)  # [index, [x, y], mark]
                        markedPointsX.append(point)
                        clusterX.append(point)
            clusterPointsX.append(clusterX)  # 每个簇为一个元素，且每个子元素含标记

    markedPointA = deleteElements(markedPointsX)  # 去除重复值 [[index, [x, y], mark],[index, [x, y], mark],...]
    clusterPointA = deleteElements(clusterPointsX)  # [[[index, [x, y], mark],[index, [x, y], mark],...],...]

    return clusterPointA, markedPointA

# ----------------------------------------------------------------------------------------------------------------------
# 以下函数用于空间叠置分析。基于向量旋转角的二维线段相交判定
# ----------------------------------------------------------------------------------------------------------------------

def readObstacle(obstacle):
    """
    从shapefile线数据中读取研究区域的空间障碍（线段）的起始点坐标，用于删除DT边列表中与障碍线段相交的边。

    输入参数：
    obstacle: 空间障碍shapefile数据，将所有需考虑的障碍（道路，河流，分水岭等）合并为一个文件，且需在vertex处打断障碍以得到起始点坐标。

    输出参数
    obstacleList: 障碍列表[[[Sx1, Sy1],[Ex1, Ey1]], ...]
    """
    obstacleList, rows, fields = [], arcpy.SearchCursor(obstacle), arcpy.ListFields(obstacle)
    start, end = [], []
    for row in rows:  # Iterate shapefile
        for field in fields:
            if field.name == "X_START":
                S_X = row.getValue(field.name)
            if field.name == "Y_START":
                S_Y = row.getValue(field.name)
            if field.name == "X_END":
                E_X = row.getValue(field.name)
            if field.name == "Y_END":
                E_Y = row.getValue(field.name)
        start.append([S_X, S_Y])
        end.append([E_X, E_Y])
        obstacleList.append([start, end])

    if len(obstacleList) < 1:
        raise Exception("EMPTY LIST: YOU GOT AN EMPTY LIST!!!")
    return obstacleList


def vectorAngle(edge, point):
    """
     输入参数
    edge: DT边或空间障碍边 [[startx, starty], [endx, endy]]
    point: 空间障碍边或DT边的起/始点坐标[x, y]

    输出参数
    rotationAngle: edge至edge起点与point连线的旋转夹角，以正弦表示。当rotationAngle > 0 时不相交；≤ 0 时为相交
    """
    if len(edge) < 1:
        raise Exception("EMPTY_ERROR: edge is an empty list!!!")

    x1 = edge[1][0] - edge[0][0]
    y1 = edge[1][1] - edge[0][1]
    x2 = point[0] - edge[0][0]
    y2 = point[1] - edge[0][1]
    rotationAngle = x1 * y2 - x2 * y1
    return rotationAngle


def intersectTest(edge1, edge2):
    """
    计算二维空间线段edge1和edge2是否相交。1--相交；0--不相交

    输入参数
    edge1: DT边 [[startx, starty], [endx, endy]]
    edge2: 空间障碍（用线段表示）[[startX, startY], [endX, endY]]

    输出参数
    result: 判断结构，1--相交；0--不相交
    """
    ZERO = 1e-11
    if vectorAngle(edge1, edge2[0]) * vectorAngle(edge1, edge2[1]) <= ZERO and \
       vectorAngle(edge2, edge1[0]) * vectorAngle(edge2, edge1[1]) <= ZERO:  # 相交
        result = 1
    else:  # 不相交
        result = 0
    return result


def reachable(otherEdgeList, markedPoints, obstacleList, pointList):
    """
    删除与障碍相交的边，返回余下DT边列表，在根据各点的一阶领域点再次做标记。

    输入参数
    otherEdgeList: 全局其他边[[id, ida,idb,len],...] id为边编号，ida/idb为边顶点索引号
    markedPoints: 有全局聚类标识的空间点[[index, [x, y], mark],[index, [x, y], mark],...]
    obstacleList: 障碍列表[[[Sx1, Sy1],[Ex1, Ey1]], ...]
    pointList: 空间点坐标列表，[[X,Y],...]

    输出参数
    reachableEdge: 删除不可达边后的DT边，数据结构同otherEdgeList
    """
    triangleEdge = []
    for edge in otherEdgeList:  # 读取DT边的端点坐标，并存放在列表triangleEdge中，数据结构同obstacleList
        start_end = []
        # for ID in edge[1:3]:
        for point in markedPoints:
            if point[0] in edge[1:3]:
                start_end.append(pointList[edge[1]])
                start_end.append(pointList[edge[2]])
            else:
                continue
        triangleEdge.append(start_end)

    reach, reachableEdge = [], []
    for i in range(len(triangleEdge)):  # 获取可达边，存放在reachable列表[[[Sx1, Sy1],[Ex1, Ey1]], ...]
        for j in range(len(obstacleList)):
            intersect = intersectTest(triangleEdge[i], obstacleList[j])
            if intersect == 0:
                reach.append(edge)
            else:
                continue

    for e in reach:  # 调整可达边数据结构与otherEdgeList一致
        indexA = pointList.index(e[0])
        indexB = pointList.index(e[1])
        for E in otherEdgeList:
            if indexA in E[1:3] and indexB in E[1:3]:
                reachableEdge.append(E)
            else:
                continue
    return reachableEdge


# ----------------------------------------------------------------------------------------------------------------------
# 以下函数用于删除局部长边
# ......................................................................................................................

def getSubgraph(reachableEdge, clusterPointB):
    """
    用于获取删除全局长边和障碍边后的所有子图，每个子图为一个元素，每个元素包含子图所有的边/顶点。subgraphEdge, subgraphPoint列表中相同索引号
    对应同一子图。

    输入参数
    reachableEdge: 删除不可达边后的DT边，数据结构同otherEdgeList [[id, ida,idb,len],...]
    clusterPointB: [[[index, [x, y], markA, markB],[index, [x, y], markA, markB],...],...]

    输出参数
    subgraphEdge: 子图边列表[[[id, ida,idb,len],[id, ida,idb,len],...],[...],...]
    subgraphPoint: 子图顶点列表[[p1,p2,p3,...],[...],...]
    """
    localEdgeSTD, subgraphEdge, subgraphPoint = [], [], []
    for i in clusterPointB:
        if i[0][-1] == "B0":
            continue
        else:
            graph, subgraphEdgeX, subgraphPointX = i, [], []
            for PNT in graph:  # 获取子图
                for pnt in PNT:
                    for e in reachableEdge:
                        if pnt[0] in e[1:3]:
                            subgraphEdgeX.append(e)
                            subgraphPointX.append(e[1])
                            subgraphPointX.append(e[2])
                edges = deleteElements(subgraphEdgeX)  # 子图边
                points = deleteElements(subgraphPointX)  # 子图顶点
                subgraphEdge.append(edges)
                subgraphPoint.append(points)
    return subgraphEdge, subgraphPoint


def deleteLocalLongEdge(vertexPoints, subgraphEdge, subgraphPoint):
    """
    用于删除局部场边，并返回余下的DT边列表

    输入参数
    vertexPoints: points coordinate with ID. [[id1,[x1,y1]],...]，不含重复点
    subgraphEdge: 子图边列表[[[id, ida,idb,len],[id, ida,idb,len],...],[...],...]
    subgraphPoint: 子图顶点列表[[p1,p2,p3,...],[...],...]

    输出参数
    localEdge: 删除局部长边后DT边列表。[[id, ida,idb,len],...]
    """
    localEdgeX = []
    for point in vertexPoints:
        edge = []
        for i in range(len(subgraphPoint)):
            if point[0] in subgraphPoint[i]:  # 子图的所有顶点[p1,p2,p3,...]
                graphEdge = subgraphEdge[i]  # 子图的所有边[[id, ida,idb,len],[id, ida,idb,len],...]
                for E in graphEdge:  # 获取子图的边长变异
                    edge.append(E[-1])
                localSTD = np.std(edge)

                # 计算子图顶点的二阶邻域边长均值
                _, firstOrderPoint = getNeighbourhood1(subgraphPoint[i], subgraphEdge[i])
                mean2 = getNeighbourhood2(subgraphPoint[i], subgraphEdge[i], firstOrderPoint)

                cutValueList = []
                for a in subgraphPoint[i]:  # 获取子图个顶点的约束准则，并生成列表[[pointIndexID, value],...]
                    for b in mean2:
                        if a[0] == b[0]:
                            cutValue = b[1] + localSTD  # 子图个顶点的约束准则列表
                        else:
                            continue
                        cutValueList.append([a[0], cutValue])

                for p in subgraphPoint[i]:  # 删除局部长边
                    for e in subgraphEdge[i]:
                        if p[0] in e[1:3]:
                            length = e[-1]
                            for value in cutValueList:
                                if p[0] == value[0] and length < value[1]:
                                    localEdgeX.append(e)
                                else:
                                    continue
                        else:
                            continue
            else:
                continue

    localEdge = deleteElements(localEdgeX)
    return localEdge


# ......................................................................................................................
# 以下函数用于最长边限定（不考虑颈、链问题），考虑到农村地区的自然社会特点，将边长上限设定为300米
# ......................................................................................................................

def lengthConstraint(localEdge, constraint):
    """
    用于限制边的长度，超过限定值得边将被打断。

    输入参数
    localEdge: 删除局部长边后DT边列表。[[id, ida,idb,len],...]
    constraint: DT边限制长度，米。

    输出参数
    unrestrictedEdge: 删除限制边后的DT边。[[id, ida,idb,len],...]
    """
    unrestrictedEdge = []
    for i in localEdge:
        if i[-1] < constraint:
            unrestrictedEdge.append(i)
    return unrestrictedEdge


# ----------------------------------------------------------------------------------------------------------------------
# 以下函数用于ArcGIS界面的可视化
# ----------------------------------------------------------------------------------------------------------------------
def createShapeFile(markedPoint, spatialRef, output):  # 无法写入ID号

    """
    根据坐标点列表创建point文件，并为其设定坐标参考。

    输入参数
    pointList: 空间点坐标列表，[[X,Y],...]
    spatialRef: 空间参考
    output: 文件输出位置及名称
    """
    point = arcpy.Point()
    pointGeometryList = []
    for i in range(len(markedPoint)):
        point.X = markedPoint[i][1][0]
        point.Y = markedPoint[i][1][1]
        point.ID = markedPoint[i][0]

        pointGeometry = arcpy.PointGeometry(point, spatialRef)
        pointGeometryList.append(pointGeometry)

    arcpy.CopyFeatures_management(pointGeometryList, output)
    return


def addMarkerFields(fileName, markedPoint):
    """
    给输出shape文件增加字段

    输入参数
    fileName: 需增加字段的文件名称及路径
    markedPoint: points coordinate with ID and markerS. [[index, [x, y], A1, B1],[index, [x, y], A2, B1],...]
    """
    arcpy.AddField_management(fileName, "ID_T", "FLOAT")
    arcpy.AddField_management(fileName, "mark1", "TEXT")  # global
    arcpy.AddField_management(fileName, "mark2", "TEXT")  # obstacle
    arcpy.AddField_management(fileName, "mark3", "TEXT")  # local
    arcpy.AddField_management(fileName, "mark4", "TEXT")  # Constraint

    counter, rows = 0, arcpy.UpdateCursor(fileName)
    for row in rows:
        row.setValue("ID_T", markedPoint[counter][0])
        row.setValue("mark1", markedPoint[counter][-4])
        row.setValue("mark2", markedPoint[counter][-3])
        row.setValue("mark3", markedPoint[counter][-2])
        row.setValue("mark4", markedPoint[counter][-1])
        rows.updateRow(row)
        counter += 1
    return


def addMarkerFields0(fileName, markedPoint):
    """
    给输出shape文件增加字段

    输入参数
    fileName: 需增加字段的文件名称及路径
    markedPoint: points coordinate with ID and markerS. [[index, [x, y], A1, B1],[index, [x, y], A2, B1],...]
    """
    arcpy.AddField_management(fileName, "ID_T", "FLOAT")
    arcpy.AddField_management(fileName, "mark4", "TEXT")  # Constraint

    counter, rows = 0, arcpy.UpdateCursor(fileName)
    for row in rows:
        row.setValue("ID_T", markedPoint[counter][0])
        row.setValue("mark4", markedPoint[counter][-1])
        rows.updateRow(row)
        counter += 1
    return