
import gmsh
import math
import os
import sys

gmsh.initialize()


path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path, 'giraffe.stl'))

angle = 40


forceParametrizablePatches = True


includeBoundary = True



curveAngle = 20
print('starting meshing...')

gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary,
                                 forceParametrizablePatches,
                                 curveAngle * math.pi / 180.)
print('Classified surfaces')

gmsh.model.mesh.createGeometry()

print('Created geometry')


s = gmsh.model.getEntities(2)
l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
gmsh.model.geo.addVolume([l])

gmsh.model.geo.synchronize()

print('Synchronised geometry')


f = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(f, "F", "5")
gmsh.model.mesh.field.setAsBackgroundMesh(f)


gmsh.model.mesh.generate(3)

print('Generated mesh')
gmsh.write('giraffe.msh')


if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
