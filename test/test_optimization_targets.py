
import dolfin as df

from pulse_adjoint import LVTestPatient


from pulse_adjoint.setup_parameters import (setup_adjoint_contraction_parameters,
                                            setup_general_parameters)

from pulse_adjoint.optimization_targets import VolumeTarget, RegionalStrainTarget



setup_general_parameters()
patient = LVTestPatient()


def get_dummy_displacement():
    params = setup_adjoint_contraction_parameters()

    from pulse_adjoint.setup_optimization import make_solver_params
    solver_parameters, pressure, paramvec= make_solver_params(params, patient)
    
    from pulse_adjoint.lvsolver import LVSolver
    solver = LVSolver(solver_parameters)
    
    from pulse_adjoint.iterate import iterate_pressure
    iterate_pressure(solver, 1.0, pressure)
    u,p = solver.get_state().split(deepcopy=True)
    return u
    
def main():
    V = df.VectorFunctionSpace(patient.mesh, "CG", 2)
    u0 = df.Function(V)
    u1 = get_dummy_displacement()

    V0 = df.VectorFunctionSpace(patient.mesh, "CG", 1)
    
    
    
    dS = df.Measure("exterior_facet",
                 subdomain_data = patient.ffun,
                 domain = patient.mesh)(patient.markers["ENDO"][0])

    
    basis = {}
    for l in ["circumferential", "radial", "longitudinal"]:
        basis[l] = getattr(patient, l)
        
    dX = df.Measure("dx", subdomain_data=patient.sfun, domain=patient.mesh)
    nregions = len(set(patient.sfun.array()))


    for u in [u0, u1]:
        for approx in ["project", "interpolate", "original"]:


            # ui = u0
            ui = u1
        
            if approx == "interpolate":
                u_int = df.interpolate(df.project(ui, V),V0)
            
            elif approx == "project":
                u_int = df.project(ui, V0)

            else:
                u_int = ui

                       
            F_ref = df.grad(u_int) + df.Identity(3)
            

            print "\nApprox = {}:".format(approx)
            target_vol = VolumeTarget(patient.mesh, dS, "LV", approx)
            target_vol.set_target_functions()
            target_vol.assign_simulated(u)
            
            vol = target_vol.simulated_fun.vector().array()[0]
            print "Volume = ", vol


            target_strain = RegionalStrainTarget(patient.mesh,
                                                 basis, 
                                                 dX,
                                                 nregions = nregions,
                                                 tensor = "gradu",
                                                 F_ref = F_ref,
                                                 approx=approx, map_strain =True)

            target_strain.set_target_functions()
            target_strain.assign_simulated(u)

            strain = [target_strain.simulated_fun[i].vector().array() \
                      for i in range(nregions)]
            print "Regional strain = ", strain
        


if __name__ == "__main__":
    main()
