# -*- coding: utf-8 -*-

# **********************************************************************************************************************
# This file is a Spatial point Pattern Analysis Algorithm, and is used for spatial point clustering analysis.
# And it could be used as assist tool to planning decision of decentralized sewage treatment facilities. This model
# contains mainly three parts, they are points trend analysis, point cluster analysis and spatial visualization.
# This algorithm

#
# Author: Yuansheng Huang
# Date: 2019.09.24
# Version: V 0.1
# Note: In V 0.1 we use 2d Euclidean distance to represent the distance between two points.

# Reference: Clark and Evans, 1954; Gao, 2013

# **********************************************************************************************************************

# general import
import gc
import os
import sys

from functions import *

pythonScriptPath = "D:/sourceCode/buildingsClusterAnalysis/pythonFiles01"

gc.disable()
pythonPath = os.getcwd()
sys.path.append(pythonPath)
sys.path.append(pythonScriptPath)

# ======================================================================================================================
# 调用ArcGIS界面输入
# ======================================================================================================================
arcpy.env.overwriteOutput = True
buildings = sys.argv[1]  # Building shape file TODO 整合时需修改
studyArea = sys.argv[2]
obstacleFile = sys.argv[3]
restrain = int(sys.argv[4])  # 管道长度约束[米]
outputFolder = sys.argv[5]

outputFile = outputFolder + "/" + "Cluster.shp"
addFiledFile = outputFile  # sys.argv[5] + ".shp"

# ----------------------------------------------------------------------------------------------------------------------
# 空间点分布模式判定
# ----------------------------------------------------------------------------------------------------------------------
pointList, spatialRef = readSpatialPoints(buildings)  # 读取空间点及输入文件的

distanceList = nearestDistance(pointList)
area = float(readArea(studyArea))  # 读取研究区域面积
index, z_test = NNI(pointList, distanceList, area)
indexList, coordinateList, _, vertexPoints = delaunayTriangle(pointList)  # 核实未使用参数

# 输出空间点集分布趋势
arcpy.AddMessage(" ")
arcpy.AddMessage("************************************")
arcpy.AddMessage("Points spatial cluster analysis was successfully calculated!!")
arcpy.AddMessage("NNI index: " + str(index))
arcpy.AddMessage("Z test: " + str(z_test))
arcpy.AddMessage("************************************")

# 开始空间点集聚类分析

arcpy.AddMessage(" ")
arcpy.AddMessage("====================================")
arcpy.AddMessage("Ready for cluster module...")
arcpy.AddMessage("====================================")
arcpy.AddMessage(" ")

_, _, edgeList = getTriangleEdge(indexList, coordinateList)

if index >= 1:  # 空间点集呈均匀(>1)/随机分布(=1)
    arcpy.AddMessage("Random distribution OR Uniform distribution (NNI >= 1)")
    arcpy.AddMessage("Skip cluster analysis module and perform edge length limit analysis!!!")

    # 删除障碍不可达边
    obstacleList = readObstacle(obstacleFile)
    reachableEdge = reachable(edgeList, obstacleList, pointList)
    markO = "O"
    _, markedPointO, _ = cluster(vertexPoints, reachableEdge, markO)
    arcpy.AddMessage("Unreachable edges were deleted !!!")

    # 删除限制长边 markedPointO
    unrestrictedEdge = lengthConstraint(edgeList, restrain)
    markC = "C"
    _, markedPointC, _ = cluster(markedPointO, unrestrictedEdge, markC)
    arcpy.AddMessage("Restricted edges were deleted !!!")

    # 结果输出
    createShapeFile(vertexPoints, spatialRef, outputFile)
    addMarkerFields0(outputFile, vertexPoints)
    arcpy.AddMessage("********************************")
    arcpy.AddMessage("Edge length limit analysis was successfully performed!!!")

elif index < 1:  # 空间点集呈聚集分布
    arcpy.AddMessage("Spatial points is aggregated, perform cluster analysis Module!!!")

    # 删除全局长边后的第一阶段聚类
    globalCutList = globalCut(vertexPoints, edgeList)
    otherEdgeList, _ = deleteLongEdge(edgeList, globalCutList)
    markG = "G"
    _, markedPointsG, _ = cluster(vertexPoints, otherEdgeList, markG)
    arcpy.AddMessage("Global long edges were deleted !!!")

    # 删除局部长边边后的第二阶段聚类
    subEdge, subgraphPoint = getSubgraph(otherEdgeList)
    localEdge = deleteLocalLongEdge(subEdge, subgraphPoint)
    markL = "L"
    _, markedPointL, _ = cluster(markedPointsG, localEdge, markL)
    arcpy.AddMessage("Local long edges were deleted !!!")

    # 删除限制长边后的第三阶段聚类
    unrestrictedEdge = lengthConstraint(otherEdgeList, restrain)
    markC = "C"
    _, markedPointC, _ = cluster(markedPointL, unrestrictedEdge, markC)
    arcpy.AddMessage("Restricted edges were deleted !!!")

    obstacleList = readObstacle(obstacleFile)
    reachableEdge = reachable(unrestrictedEdge, obstacleList, pointList)
    markO = "O"
    clusterPointO, markedPointO, _ = cluster(markedPointL, reachableEdge, markO)
    arcpy.AddMessage("Unreachable edges were deleted !!!")

    # 结果输出
    createShapeFile(vertexPoints, spatialRef, outputFile)
    addMarkerFields(outputFile, vertexPoints)
    arcpy.AddMessage("************************************")

arcpy.AddMessage("Point Spatial Cluster successfully performed!!!")
arcpy.AddMessage("********************************")
# ----------------------------------------------------------------------------------------------------------------------

