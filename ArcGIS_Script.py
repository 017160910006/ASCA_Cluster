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
buildings = sys.argv[1]
studyArea = sys.argv[2]
obstacleFile = sys.argv[3]
restrain = int(sys.argv[4])
outputFolder = sys.argv[5]

outputFile = outputFolder + "/" + "Cluster.shp"
addFiledFile = outputFile


pointList, spatialRef = readSpatialPoints(buildings) 

distanceList = nearestDistance(pointList)
area = float(readArea(studyArea))
index, z_test = NNI(pointList, distanceList, area)
indexList, coordinateList, _, vertexPoints = delaunayTriangle(pointList)

arcpy.AddMessage(" ")
arcpy.AddMessage("************************************")
arcpy.AddMessage("Points spatial cluster analysis was successfully calculated!!")
arcpy.AddMessage("NNI index: " + str(index))
arcpy.AddMessage("Z test: " + str(z_test))
arcpy.AddMessage("************************************")

arcpy.AddMessage(" ")
arcpy.AddMessage("====================================")
arcpy.AddMessage("Ready for cluster module...")
arcpy.AddMessage("====================================")
arcpy.AddMessage(" ")

_, _, edgeList = getTriangleEdge(indexList, coordinateList)

if index >= 1:
    arcpy.AddMessage("Random distribution OR Uniform distribution (NNI >= 1)")
    arcpy.AddMessage("Skip cluster analysis module and perform edge length limit analysis!!!")

    obstacleList = readObstacle(obstacleFile)
    reachableEdge = reachable(edgeList, obstacleList, pointList)
    markO = "O"
    _, markedPointO, _ = cluster(vertexPoints, reachableEdge, markO)
    arcpy.AddMessage("Unreachable edges were deleted !!!")

    unrestrictedEdge = lengthConstraint(edgeList, restrain)
    markC = "C"
    _, markedPointC, _ = cluster(markedPointO, unrestrictedEdge, markC)
    arcpy.AddMessage("Restricted edges were deleted !!!")

    createShapeFile(vertexPoints, spatialRef, outputFile)
    addMarkerFields0(outputFile, vertexPoints)
    arcpy.AddMessage("********************************")
    arcpy.AddMessage("Edge length limit analysis was successfully performed!!!")

elif index < 1:
    arcpy.AddMessage("Spatial points is aggregated, perform cluster analysis Module!!!")

    globalCutList = globalCut(vertexPoints, edgeList)
    otherEdgeList, _ = deleteLongEdge(edgeList, globalCutList)
    markG = "G"
    _, markedPointsG, _ = cluster(vertexPoints, otherEdgeList, markG)
    arcpy.AddMessage("Global long edges were deleted !!!")

    subEdge, subgraphPoint = getSubgraph(otherEdgeList)
    localEdge = deleteLocalLongEdge(subEdge, subgraphPoint)
    markL = "L"
    _, markedPointL, _ = cluster(markedPointsG, localEdge, markL)
    arcpy.AddMessage("Local long edges were deleted !!!")

    unrestrictedEdge = lengthConstraint(otherEdgeList, restrain)
    markC = "C"
    _, markedPointC, _ = cluster(markedPointL, unrestrictedEdge, markC)
    arcpy.AddMessage("Restricted edges were deleted !!!")

    obstacleList = readObstacle(obstacleFile)
    reachableEdge = reachable(unrestrictedEdge, obstacleList, pointList)
    markO = "O"
    clusterPointO, markedPointO, _ = cluster(markedPointL, reachableEdge, markO)
    arcpy.AddMessage("Unreachable edges were deleted !!!")

    createShapeFile(vertexPoints, spatialRef, outputFile)
    addMarkerFields(outputFile, vertexPoints)
    arcpy.AddMessage("************************************")

arcpy.AddMessage("Point Spatial Cluster successfully performed!!!")
arcpy.AddMessage("********************************")
# ----------------------------------------------------------------------------------------------------------------------

