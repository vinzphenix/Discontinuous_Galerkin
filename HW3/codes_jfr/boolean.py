import gmsh
import sys

gmsh.initialize()

gmsh.model.add("t4")

rect = gmsh.model.occ.add_rectangle(0, 0, 0, 0.5, 1, 0)
disk = gmsh.model.occ.add_disk(0.5, 0.5, 0, 0.2, 0.2, 1000)
mm = gmsh.model.occ.cut([(2, rect)], [(2, disk)])

gmsh.model.mesh.set_size_callback(lambda *args: .1)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(3)
gmsh.option.setNumber("Mesh.Nodes", 1)
gmsh.write("t4.msh")
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
