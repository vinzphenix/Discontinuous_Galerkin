import gmsh
import sys

gmsh.initialize()

gmsh.model.add("t4")

X = 1.0
Y = 2.0
lc = 0.03 * (X * X + Y * Y) ** 0.5

rect = gmsh.model.occ.add_rectangle(0, 0, 0, X, Y, 0)

gmsh.model.mesh.set_size_callback(lambda *args: lc)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(3)
gmsh.option.setNumber("Mesh.Nodes", 1)
gmsh.write("t4.msh")
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
