import gmsh
import sys


def create_square(filename, elemSizeRatio):
    gmsh.initialize()
    gmsh.model.add("t1")
    # lc = 0.1
    lc = elemSizeRatio / 1.
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(3, 2, 2)
    gmsh.model.geo.addLine(3, p4, 3)
    gmsh.model.geo.addLine(4, 1, p4)
    gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


def create_circle(filename, elemSizeRatio):
    gmsh.initialize()
    gmsh.model.add("t1")
    lc = elemSizeRatio * 100.
    # lc = 5.

    gmsh.model.geo.addPoint(50., 0., 0, lc, 1)
    gmsh.model.geo.addPoint(100., 50., 0, lc, 2)
    gmsh.model.geo.addPoint(50., 100., 0, lc, 3)
    gmsh.model.geo.addPoint(0., 50., 0, lc, 4)
    gmsh.model.geo.addPoint(50., 50., 0, lc, 5)

    gmsh.model.geo.addCircleArc(1, 5, 3, 1)
    gmsh.model.geo.addCircleArc(3, 5, 1, 2)

    gmsh.model.geo.addCurveLoop([1, 2], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


def create_hole(filename, elemSizeRatio):
    gmsh.initialize()
    gmsh.model.add("hole")

    rect = gmsh.model.occ.add_rectangle(0, 0, 0, 1.5, 1., 0)
    disk = gmsh.model.occ.add_disk(1.5, 0.5, 0, 0.2, 0.2, 1000)
    mm = gmsh.model.occ.cut([(2, rect)], [(2, disk)])

    gmsh.model.mesh.set_size_callback(lambda *args: elemSizeRatio * 1.5)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(3)
    gmsh.option.setNumber("Mesh.Nodes", 1)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


def create_rectangle(filename, elemSizeRatio):
    gmsh.initialize()
    gmsh.model.add("rectangle")

    X = 1.0
    Y = 2.0
    # lc = 0.03 * (X * X + Y * Y) ** 0.5

    rect = gmsh.model.occ.add_rectangle(0, 0, 0, X, Y, 0)
    gmsh.model.mesh.set_size_callback(lambda *args: elemSizeRatio / 1.)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(3)
    gmsh.option.setNumber("Mesh.Nodes", 1)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


create_square("square_low.msh", elemSizeRatio=1./10.)
# create_circle("circle_h8.msh", elemSizeRatio=8./100.)
# create_hole("hole.msh", elemSizeRatio=1./20.)
# create_rectangle("rectangle.msh", elemSizeRatio=1./20.)
