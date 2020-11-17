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

    return indexList, coordinateList, DT, vertexPoints


def getID(a, b):
    """
    根据DT顶点索引生成DT边ID号，只获取一条边的ID号，而非ID列表。
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
    """
    edgeListTriangle, edgeListAll, edgeList = [], [], []
    for i in range(len(indexList)):
        indexI = indexList[i]
        indexC = coordinateList[i]
        a, b, c = indexI[0], indexI[1], indexI[2]
        m, l, n = indexC[0], indexC[1], indexC[2]
        ID1, ID2, ID3 = getID(a, b), getID(a, c), getID(b, c) 
        ID1A, ID1B = getIndex(ID1)
        ID2A, ID2B = getIndex(ID2)
        ID3A, ID3B = getIndex(ID3)
        len1 = math.hypot(m[0] - l[0], m[1] - l[1])
        len2 = math.hypot(m[0] - n[0], m[1] - n[1])
        len3 = math.hypot(l[0] - n[0], l[1] - n[1])
        edgeListTriangle.append([[ID1, ID1A, ID1B, len1], [ID2, ID2A, ID2B, len2], [ID3, ID3A, ID3B, len3]])

    for j in range(len(edgeListTriangle)):
        edge = edgeListTriangle[j]
        for k in range(len(edge)):
            edgeListAll.append(edge[k])
    edgeList = deleteElements(edgeListAll)

    return edgeListTriangle, edgeListAll, edgeList


def getNeighbourhood1(vertexPoints, edgeList):
    """
    用于获取vertex的一阶邻域顶点，并返回各点的一阶邻域边长均值（用于计算二阶邻域边长均值）
    """
    neighbourhood1AVG = []
    firstOrderPoint = []
    for i in vertexPoints:  # iteration
        length1 = []
        firstPointX = []
        for j in edgeList:
            if i[0] in j[1:3]:
                length1.append(j[-1])
                firstPointX.append(j[1])
                firstPointX.append(j[2])
            else:
                continue
        firstPoint = deleteElements(firstPointX)

        firstPoint.append(i[0])
        neighbourhood1AVG.append([i[0], np.mean(length1)])
        firstOrderPoint.append(firstPoint)
    return neighbourhood1AVG, firstOrderPoint


def getNeighbourhood2(vertexPoints, edgeList, firstOrderPoint):
    """
    获取vertexPoints列表各点的二阶邻域边长均值，任意点P的一阶邻域点的各条边包含点P二阶邻域边和点P一阶邻域三角形的所有边。所以计算二阶
    邻域时，先求点P所有一阶邻域点的所有边，再取相互不重复的边即可。
    """
    neighbourhood2AVG = []
    for i in vertexPoints:
        secondPoint = []
        edge = []
        length2 = []
        for j in firstOrderPoint:
            if i[0] == j[-1]:
                secondPoint = j[:-1]
        for a in secondPoint:
            for b in edgeList:
                if a in b:
                    edge.append([a, b])
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
    for i in edgeList:
        edge.append(i[-1])
    edgeAVG = np.mean(edge)
    edgeSTD = np.std(edge)
    return edgeAVG, edgeSTD


def globalCut(vertexPoints, edgeList):
    """
    获取各顶点的全局约束值
    """
    globalAVG, globalSTD = calcGlobalStatistic(edgeList)
    firstOrderEdgeAVG, _ = getNeighbourhood1(vertexPoints, edgeList)
    globalCutList = []
    for i in vertexPoints:
        mean1 = 0
        for j in firstOrderEdgeAVG:
            if i[0] == j[0]:
                mean1 = j[1]
        cutValue = globalAVG + 1 * (globalAVG / mean1) * globalSTD
        globalCutList.append([i[0], cutValue])
    return globalCutList


def deleteLongEdge(edgeList, globalCutList):
    """
    删除全局长边，并返回全局长边和全局其他边列表
    """
    otherEdgeListX, longEdgeListX = [], []
    for i in globalCutList:
        for j in edgeList:
            if i[0] in j and i[1] > j[-1]:
                otherEdgeListX.append(j)
            if i[0] in j and i[1] <= j[-1]:
                longEdgeListX.append(j)

    otherEdgeList = deleteElements(otherEdgeListX)
    longEdgeList = deleteElements(longEdgeListX)
    return otherEdgeList, longEdgeList


def getIsolatedPoints(vertexPoints, otherEdgeList, mark):
    """
    用于获取空间点集中的孤立点，并给每个点增加标记，在cluster函数中调用。
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


def cluster(vertexPoints, otherEdgeList, mark):  # todo 扁平化设计  黄源生 20191007
    """
    根据全局其他边，初步聚类，给个点添加簇标号，并以列表的形式返回（嵌套列表，在之前vertexPoint的基础上每个元素的末尾添加标识。
    """
    isolatePointList, otherPointsList = getIsolatedPoints(vertexPoints, otherEdgeList, mark)

    length2 = len(isolatePointList)
    num2 = str(length2)
    E2 = str(isolatePointList)
    arcpy.AddMessage("isolatePointList number: " + num2 + E2)

    length2 = len(otherPointsList)
    num2 = str(length2)
    E2 = str(otherPointsList)
    arcpy.AddMessage("otherPointsList number: " + num2 + E2)

    _, firstOrder = getNeighbourhood1(otherPointsList, otherEdgeList)
    firstOrderPoints = [i[:-1] for i in firstOrder]

    mergedPoints = aggregation(firstOrderPoints)

    operateVertexPoints = copy.deepcopy(vertexPoints)
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


def cluster1(vertexPoints, otherEdgeList, mark):
    """
    根据全局其他边，初步聚类，给个点添加簇标号，并以列表的形式返回（嵌套列表，在之前vertexPoint的基础上每个元素的末尾添加标识。
    """
    marker1 = mark + str(0)
    isolate, otherPoint, pntX, testPoint = [], [], [], copy.deepcopy( vertexPoints)
    pnt1, pnt2 = [], []
    for pt in testPoint:
        for l in otherEdgeList:
            if pt[0] not in l[1:3]:
                pnt1.append(pt)
            else:
                pnt2.append(pt)
        isolate = deleteElements(pnt1)
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
    for PI in firstOrderPoints:
        clusterX, length = [], len(PI)
        marker2 = mark + str(length)
        if PI == [0]:
            continue
        elif PI != [0]:
            for E in PI:  # get point index
                for PC in vertexPoints:
                    point = PC
                    if E == point[0]:
                        point.append(marker2)
                        markedPointsX.append(point)
                        clusterX.append(point)
            clusterPointsX.append(clusterX)
    markedPointA = deleteElements(markedPointsX)
    clusterPointA = deleteElements(clusterPointsX)

    return clusterPointA, markedPointA


def readObstacle(obstacle):
    """
    从shapefile线数据中读取研究区域的空间障碍（线段）的起始点坐标，用于删除DT边列表中与障碍线段相交的边。
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
    """
    ZERO = 1e-11
    if vectorAngle(edge1, edge2[0]) * vectorAngle(edge1, edge2[1]) <= ZERO and \
       vectorAngle(edge2, edge1[0]) * vectorAngle(edge2, edge1[1]) <= ZERO:
        result = 1
    else:  # 不相交
        result = 0
    return result


def reachable(otherEdgeList, markedPoints, obstacleList, pointList):
    """
    删除与障碍相交的边，返回余下DT边列表，在根据各点的一阶领域点再次做标记。
    """
    triangleEdge = []
    for edge in otherEdgeList:
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
    for i in range(len(triangleEdge)):
        for j in range(len(obstacleList)):
            intersect = intersectTest(triangleEdge[i], obstacleList[j])
            if intersect == 0:
                reach.append(edge)
            else:
                continue

    for e in reach:
        indexA = pointList.index(e[0])
        indexB = pointList.index(e[1])
        for E in otherEdgeList:
            if indexA in E[1:3] and indexB in E[1:3]:
                reachableEdge.append(E)
            else:
                continue
    return reachableEdge


def getSubgraph(reachableEdge, clusterPointB):
    """
    用于获取删除全局长边和障碍边后的所有子图，每个子图为一个元素，每个元素包含子图所有的边/顶点。subgraphEdge, subgraphPoint列表中相同索引号
    对应同一子图。
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
                edges = deleteElements(subgraphEdgeX)
                points = deleteElements(subgraphPointX)
                subgraphEdge.append(edges)
                subgraphPoint.append(points)
    return subgraphEdge, subgraphPoint


def deleteLocalLongEdge(vertexPoints, subgraphEdge, subgraphPoint):
    """
    用于删除局部场边，并返回余下的DT边列表
    """
    localEdgeX = []
    for point in vertexPoints:
        edge = []
        for i in range(len(subgraphPoint)):
            if point[0] in subgraphPoint[i]:
                graphEdge = subgraphEdge[i]
                for E in graphEdge:
                    edge.append(E[-1])
                localSTD = np.std(edge)

                _, firstOrderPoint = getNeighbourhood1(subgraphPoint[i], subgraphEdge[i])
                mean2 = getNeighbourhood2(subgraphPoint[i], subgraphEdge[i], firstOrderPoint)

                cutValueList = []
                for a in subgraphPoint[i]:
                    for b in mean2:
                        if a[0] == b[0]:
                            cutValue = b[1] + localSTD
                        else:
                            continue
                        cutValueList.append([a[0], cutValue])

                for p in subgraphPoint[i]:
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


def lengthConstraint(localEdge, constraint):
    """
    用于限制边的长度，超过限定值得边将被打断。
    """
    unrestrictedEdge = []
    for i in localEdge:
        if i[-1] < constraint:
            unrestrictedEdge.append(i)
    return unrestrictedEdge


def createShapeFile(markedPoint, spatialRef, output):
    """
    根据坐标点列表创建point文件，并为其设定坐标参考。
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
