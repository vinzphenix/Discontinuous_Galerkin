import gmsh
import sys
import numpy as np

gmsh.initialize()

order = 3
gmsh.open("t4.msh")
gmsh.model.mesh.setOrder(order)

gmsh.model.mesh.createEdges()

elementType = gmsh.model.mesh.getElementType("triangle", order)

elementTags, elementNodeTags = gmsh.model.mesh.getElementsByType(elementType)

edgeNodes = gmsh.model.mesh.getElementEdgeNodes(elementType).reshape(-1, order + 1)

print(edgeNodes)

edgeTags, edgeOrientations = gmsh.model.mesh.getEdges(edgeNodes[:, :2].flatten())

edges2Elements = {}

for i in range(len(edgeTags)):  # 3 edges per triangle
    if not edgeTags[i] in edges2Elements:
        edges2Elements[edgeTags[i]] = [elementTags[i // 3]]
    else:
        print(edgeTags[i])
        edges2Elements[edgeTags[i]].append(elementTags[i // 3])

print(edges2Elements)
