
#from patient_data import FullPatient # Put example mesh in the folder later
#import dolfin as df
#import os
#path = os.path.dirname(os.path.abspath(__file__))
#meshpath = "/".join([path, "demo_mesh.xml"])
#print meshpath
#mesh = df.Mesh(str(meshpath))

def demo_mesh_cut():
    try:
        import dolfinplot as dfp
    except:
        raise IOError("Misssing dolfinplot. git clone https://bitbucket.org/finsberg/dolfinplot.git")
    vtkmesh = dfp.VTK_DolfinMesh(mesh)
    vtkmesh.Clip(origin = (4.5, 0.0, 0.0), normal = (0.0,-1.0,1.0), overlay=True)
    vtkmesh.Render(view = "side", dpi =1000, size = (1200,800))
    vtkmesh.Show()   
    
def demo_mesh(mesh):
    try:
        import dolfinplot as dfp
    except:
        raise IOError("Misssing dolfinplot. git clone https://bitbucket.org/finsberg/dolfinplot.git")
    vtkmesh = dfp.VTK_DolfinMesh(mesh)
    vtkmesh.SetRepresentation("surface_w_edges")
    vtkmesh.SetEdgeColor((0,0,0))
    focal = (mesh.coordinates().T[0].max()) / 2.0
    vtkmesh.Render(view = "side", dpi =1000, size = (1200,800), focal=focal)
    #vtkmesh.Show()
    return vtkmesh

def demo_cellfunction(mesh):
    try:
        import dolfinplot as dfp
    except:
        raise IOError("Misssing dolfinplot. git clone https://bitbucket.org/finsberg/dolfinplot.git")
    import dolfin as df
    cfun = df.MeshFunction("size_t", mesh, 3, mesh.domains())
    V = df.FunctionSpace(mesh, "DG", 0)
    f = df.Function(V)
    f.vector()[:] = cfun.array()
    
    vtkfun = dfp.VTK_DolfinScalar(f)
    vtkfun.SetEdgeColor((0,0, 0))
    focal = (mesh.coordinates().T[0].max()) / 2.0
    vtkfun.Render(view = "side", dpi =300, size = (1200,800), focal=focal)
    #vtkfun.Show()
    return vtkfun
    
def demo_facetfunction(mesh):
    try:
        import dolfinplot as dfp
    except:
        raise IOError("Misssing dolfinplot. git clone https://bitbucket.org/finsberg/dolfinplot.git")
    import dolfin as df
    ffun = df.MeshFunction("size_t", mesh, 2, mesh.domains())
    bmesh = df.BoundaryMesh(mesh, "exterior")

    Vb = df.FunctionSpace(bmesh, "DG", 0)
    fb = df.Function(Vb)
    mapping = bmesh.entity_map(2)
   
    for cell in df.cells(bmesh):
        # fb.vector()[cell.index()] = ffun[mapping[cell.index()]]
        if ffun[mapping[cell.index()]] == 10:
            fb.vector()[cell.index()] = 1
        elif ffun[mapping[cell.index()]] == 30:
            fb.vector()[cell.index()] = 2
        elif ffun[mapping[cell.index()]] == 40:
            fb.vector()[cell.index()] = 0
            
    vtkfun = dfp.VTK_DolfinScalar(fb)
    vtkfun.SetEdgeColor((0,0, 0))
    focal = (mesh.coordinates().T[0].max()) / 2.0
    vtkfun.Render(view = "side", dpi =300, size = (1200,800), focal = focal)
    #vtkfun.Show()
    return vtkfun
    

def demo_scalarfunction(mesh, sf):
    try:
        import dolfinplot as dfp
    except:
        raise IOError("Misssing dolfinplot. git clone https://bitbucket.org/finsberg/dolfinplot.git")
    import dolfin as df
    #fun = df.Expression("x[0]")
    V =  df.FunctionSpace(mesh, "Lagrange", 1)
    sf = df.project(sf, V)
    g = df.interpolate(sf, V)
    
    vtkfun = dfp.VTK_DolfinScalar(g)
    vtkfun.SetEdgeColor((0,0, 0))
    vtkfun.SetColorBar(title = ' ')
    focal = (mesh.coordinates().T[0].max()) / 2.0
    vtkfun.Render(view = "side", dpi =300, size = (1200,800), focal = focal)
    #vtkfun.Show()
    return vtkfun

def demo_vectorfunction():
    try:
        import dolfinplot as dfp
    except:
        raise IOError("Misssing dolfinplot. git clone https://bitbucket.org/finsberg/dolfinplot.git")
    import dolfin as df
    fun = df.Expression(("1", "0","0"))
    element = df.VectorElement(family = "Quadrature",
                               cell = mesh.ufl_cell(),
                               degree = 2,
                               quad_scheme="default")
    V =  df.FunctionSpace(mesh, element)
    # V = df.VectorFunctionSpace(mesh, "CG", 1)
    g = df.interpolate(fun, V)
    
    vtkvector = dfp.VTK_DolfinVector(g, plot_mesh=False)
    vtkvector.SetGlyph("arrow", tipLength = 0.15, tipRadius = 0.05, shaftRadius = 0.03)
    vtkvector.SetColor((1,0,0))
    vtkvector.Render(view = "side", dpi =300, size = (1200,800))
    vtkvector.Show()

def demo_displacement(mesh, u):
    try:
        import dolfinplot as dfp
    except:
        raise IOError("Misssing dolfinplot. git clone https://bitbucket.org/finsberg/dolfinplot.git")
    import dolfin as df
    vtkmesh_old = dfp.VTK_DolfinMesh(mesh)
    vtkmesh_old.SetOpacity(0.5)    
    
    #fun1 = df.Expression(("0", "sin(x[1])","0"))
    #V1 = df.VectorFunctionSpace(mesh, "CG", 1)
    #d1 = df.interpolate(fun1, V1)
    
    V=df.VectorFunctionSpace(mesh, "CG", 1)
    d = df.interpolate(u, V)

    movedmesh = df.Mesh(mesh)
    df.ALE.move(movedmesh, d)
    #V3 = df.VectorFunctionSpace(movedmesh, "CG", 1)
    #d3 = df.interpolate(fun1, V3)
    
    #fun2 = df.Expression("x[0]")
    V2 =  df.FunctionSpace(movedmesh, "Lagrange", 1)
    #d2 = df.interpolate(fun1, V2)
    
    d_length = df.project(df.sqrt(d**2), V2)    
    
    #vtkmesh = dfp.VTK_DolfinMesh(movedmesh)
    vtkmesh = dfp.VTK_DolfinScalar(d_length)
    vtkmesh.SetRepresentation("surface_w_edges")
    vtkmesh.SetEdgeColor((0,0, 0))
    vtkmesh.renderer.AddActor(vtkmesh_old.actor)
    vtkmesh.SetColorBar(title = ' ')
    focal = (mesh.coordinates().T[0].max()) / 2.0
    vtkmesh.Render(view = "side", dpi =300, size = (1200,800), focal=focal)
    print 'her5'
    return vtkmesh
    # VTK way ?

    