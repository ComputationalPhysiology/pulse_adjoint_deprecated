from setup_optimization import setup_adjoint_contraction_parameters, setup_general_parameters
from run_optimization import run_passive_optimization, run_active_optimization
from adjoint_contraction_args import *
from numpy_mpi import *
from utils import passive_inflation_exists, contract_point_exists,  Text, pformat
from dolfin_adjoint import adj_reset


def initialize_patient_data(patient_parameters, synth_data):

    logger.info(Text.blue("Initialize patient data"))
    from patient_data import Patient
    
    patient = Patient(**patient_parameters)

    # if args_full.use_deintegrated_strains:
        # patient.load_deintegrated_strains(STRAIN_FIELDS_PATH)

    if synth_data:
        patient.passive_filling_duration = SYNTH_PASSIVE_FILLING
        patient.num_contract_points =  NSYNTH_POINTS + 1
        patient.num_points = SYNTH_PASSIVE_FILLING + NSYNTH_POINTS + 1

    return patient


def main(params):

    setup_general_parameters()
    

    logger.info(Text.blue("Start Adjoint Contraction"))
    logger.debug(pformat(params.to_dict()))
    

    ############# GET PATIENT DATA ##################
    patient = initialize_patient_data(params["Patient_parameters"], 
                                      params["synth_data"])


    ############# RUN MATPARAMS OPTIMIZATION ##################
    
    # Make sure that we choose passive inflation phase
    params["phase"] =  PHASES[0]
    if not passive_inflation_exists(params):
        run_passive_optimization(params, patient)
        adj_reset()


    
    ################## RUN GAMMA OPTIMIZATION ###################

    # Make sure that we choose active contraction phase
    params["phase"] =  PHASES[1]
    run_active_optimization(params, patient)
   
        
if __name__ == '__main__':

    # parser = get_parser()
    # args = parser.parse_args()
    # main(args)
    
    params = setup_adjoint_contraction_parameters()
    main(params)
    
