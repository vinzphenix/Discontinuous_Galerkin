import gmsh
import sys
gmsh.initialize()
gmsh.model.add("t1")
lc = 0.1
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(2, 0, 0, lc, 2)
gmsh.model.geo.addPoint(2, 2, 0, lc, 3)
p4 = gmsh.model.geo.addPoint(0, 2, 0, lc)
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(3, 2, 2)
gmsh.model.geo.addLine(3, p4, 3)
gmsh.model.geo.addLine(4, 1, p4)
gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("../t2.msh")
if '-nopopup' not in sys.argv:
   gmsh.fltk.run()
gmsh.finalize()
