#1. Import libraries
import sys
import os
path = os.getcwd() + '/../../src/'
sys.path.append(path)

import numpy as np
import h5py

import nlopt
                                  
import p_Weighted_Head_Gradient, Permeability, Steady_State_Crafter, Density_Filter, PFLOTRAN, Volume_Percentage


if __name__ == "__main__":
  #2. create PFLOTRAN simulation object
  pflotranin = "pflotran.in"
  sim = PFLOTRAN.PFLOTRAN(pflotranin)
  
  #3. Parametrize hydraulic conductivity
  #get cell ids in the region to optimize and parametrize permeability
  #same name than in pflotran input file
  pit_ids = sim.get_region_ids("Pit")
  perm = Permeability.Permeability([5e-14,1e-10], cell_ids_to_parametrize=pit_ids, power=3)
  
  #4. Define the cost function and constrain
  #define cost function as sum of the head in the pit
  cf = p_Weighted_Head_Gradient.p_Weighted_Head_Gradient(pit_ids, invert_weighting=True)
  #define maximum volume constrains
  max_vol = Volume_Percentage.Volume_Percentage(pit_ids, 0.2)
  
  #5. Define filter
  filter = Density_Filter.Density_Filter(6) #search neighbors in a 3 m radius
  
  #6. Craft optimization problem
  #i.e. create function to optimize, initiate IO array in classes...
  crafted_problem = Steady_State_Crafter.Steady_State_Crafter(cf, sim, [perm], [max_vol], filter)
  crafted_problem.obj.adjoint.set_adjoint_problem_algo("bicgstab")
  crafted_problem.output_every_iteration(5)
  
  #7. Initialize optimizer
  #create optimize
  algorithm = nlopt.LD_MMA
  opt = nlopt.opt(algorithm, crafted_problem.get_problem_size())
  opt.set_min_objective(crafted_problem.nlopt_function_to_optimize)
  #add volume constrain
  opt.add_inequality_constraint(crafted_problem.nlopt_constrain(0), #max volume
                                0.005) #function, tolerance
  #add bound on the density parameter: p in [0,1]
  opt.set_lower_bounds(np.zeros(crafted_problem.get_problem_size(), dtype='f8')+0.001)
  opt.set_upper_bounds(np.ones(crafted_problem.get_problem_size(), dtype='f8'))
  #define stop criterion
  opt.set_ftol_rel(0.00000001)
  opt.set_maxeval(100)
  #initial guess
  p = np.random.random(crafted_problem.get_problem_size())
  p[p>0.8] = 0.9
  p[p<1] = 0.001
  
  #8. Run optimization
  try:
    p_opt = opt.optimize(p)
    sim.create_cell_indexed_dataset(p_opt, "Density parameter optimized","p_opt.h5")
  except(KeyboardInterrupt):
    sim.create_cell_indexed_dataset(crafted_problem.last_p, 
                                    "Density parameter optimized",
                                    f"p_{crafted_problem.func_eval}.h5")
  
