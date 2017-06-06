def setup_moving_mesh(state_space, newmesh):
    V = state_space.sub(0).collapse()
    u_prev = Function(V)
    u_current = Function(V)
    state = Function(state_space)

    fa = FunctionAssigner(V, state_space.sub(0))
    
    d = Function(VectorFunctionSpace(newmesh,  "CG", 2))
    
    return u_prev, u_current, state, d, fa

def setup_bullseye_sim(bullseye_mesh, fun_arr):
    V = FunctionSpace(bullseye_mesh, "DG", 0)
    dm = V.dofmap()
    sfun = MeshFunction("size_t", bullseye_mesh, 2, bullseye_mesh.domains())

    funcs = []
    for time in range(len(fun_arr)):

        fun_tmp = Function(V)
        arr = fun_arr[time]

        for region in range(17):

                vertices = []

                for cell in cells(bullseye_mesh):

                    if sfun.array()[cell.index()] == region+1:

                        verts = dm.cell_dofs(cell.index())

                        for v in verts:
                            # Find the correct vertex index 
                            if v not in vertices:
                                vertices.append(v)

                fun_tmp.vector()[vertices] = arr[region]
        funcs.append(Vector(fun_tmp.vector()))
    return funcs
    
