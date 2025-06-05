'''
Description: main file for LS model. Allows for specification of desired output
             options: 'data' (.csv files), 'graphs'
             and specification of desired run types
             options: ''

'''


import lynch_presets as ps
import lynch_simulator as sim
import lynch_icer as ic

'''
#RUN OPTIONS:
'''
#CEA-- generates output for CEA
#INTERVAL_GRAPHS-- generates graphs with outputs for each csy interval
#AGE_GRAPHS--generates graphs with outputs for each csy start age
#INTERVAL_FILES -- generates .csv files with distributions for varying csy intervals
#AGE_FILES-- generates .csv files with distributions for varying csy start age
#ALL_FILES--generates .csv files with distributions for all strategies

#----------------------------------------

def main():
    my_run_mode = 'ALL_FILES'
	# print('In main...')
    if my_run_mode == 'CEA':
        #if necessary files don't exist already, this fxn should prompt others to create them
        ic.run_CEA()
    elif my_run_mode == 'INTERVAL_GRAPHS':
        sim.graph_everything_by_int()
    elif my_run_mode == 'AGE_GRAPHS':
        sim.graph_everything_by_age()
    elif my_run_mode == 'ALL_FILES':
        for gender in ps.genders:
            sim.generate_output(gender)
    elif my_run_mode == 'AGE_FILES':
        sim.generate_output_lite_start_age()
    elif my_run_mode == 'INTERVAL_FILES':
        sim.generate_output_lite_interval()
    else:
        print('run mode does not exist')
    return()


if __name__ == "__main__":
	main()

