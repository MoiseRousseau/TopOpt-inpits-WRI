import numpy as np



class Volume_Percentage:
  """
  Function that compute the percentage of the volume occupied the material represented by
  p=1 (default) or by p=0 by setting the option volume_of_p0 to True.
  Return a negative value if the actual percentage is lower than the max percentage, and 
  positive instead
  Inputs:
  - ids_to_sum_volume: the cell ids where to calculate the percentage (default: everywhere)
  - max_volume_percentage: the maximum volume percentage (default: 0.2)
  """
  def __init__(self, ids_to_sum_volume="parametrized_cell", max_volume_percentage=0.2,
                     volume_of_p0=False):
    if isinstance(ids_to_sum_volume, str):
      if ids_to_sum_volume.lower() == "parametrized_cell":
        self.ids_to_consider = None
      else:
        print("Error! Non-recognized option for ids_to_sum_volume: " + ids_to_sum_volume)
        exit(1)
    else:
      self.ids_to_consider = ids_to_sum_volume
    self.max_v_frac = max_volume_percentage
    self.vp0 = volume_of_p0 #boolean to compute the volume of the mat p=1 (False) p=0 (True)
    
    #function inputs
    self.V = None
    
    #quantities derived from the input calculated one time
    self.initialized = False
    self.V_tot = None
    
    #function derivative for adjoint
    self.dobj_dP = 0.
    self.dobj_dmat_props = [0.]
    self.dobj_dp_partial = None
    self.adjoint = None
    
    self.output_variable_needed = ["VOLUME"] 
    self.name = "Volume"
    return
  
  
  def set_inputs(self, inputs):
    self.V = inputs[0]
    return
  
  def get_inputs(self):
    return [self.V]
    
  def set_p_to_cell_ids(self, cell_ids):
    self.p_ids = cell_ids
    return
  
  
  ### COST FUNCTION ###
  def evaluate(self,p):
    """
    Evaluate the cost function
    Return a scalar of dimension [L**3]
    """
    if not self.initialized: self.__initialize__()
    if self.vp0: p_ = 1-p
    else: p_ = p
    if self.ids_to_consider is None:
      #sum on all parametrized cell
      cf = np.sum(self.V[self.p_ids-1]*p_)/self.V_tot - self.max_v_frac
    else:
      cf = np.sum((self.V[self.ids_to_consider-1]*p_))/self.V_tot - self.max_v_frac
    return cf
  
  
  ### PARTIAL DERIVATIVES ###
  def d_objective_dP(self,p): 
    return 0.
    
  def d_objective_d_mat_props(self,p): 
    return [0.]
  
  def d_objective_dp_partial(self,p): 
    if self.dobj_dp_partial is None:
      self.dobj_dp_partial = np.zeros(len(p),dtype='f8')
    if self.vp0: factor = -1.
    else: factor = 1.
    if not self.initialized: self.__initialize__()
    if self.ids_to_consider is None:
      self.dobj_dp_partial[:] = factor * self.V[self.p_ids-1]/self.V_tot
    else:
      self.dobj_dp_partial[:] = factor * self.V[self.ids_to_consider-1]/self.V_tot
    return
  
  ### TOTAL DERIVATIVE ###
  def d_objective_dp_total(self, p, out=None):
    if out is None: out = np.zeros(len(p),dtype='f8')
    self.d_objective_dp_partial(p)
    out[:] = self.dobj_dp_partial
    return out
    
  
  ### WRAPPER ###
  def nlopt_optimize(self, p, grad):
    cf = self.evaluate(p)
    if grad.size > 0:
      self.d_objective_dp_total(p, grad)
    return cf
    
  
  ### INITIALIZER FUNCTION ###
  def __initialize__(self):
    """
    Initialize the derived quantities from the inputs that must be compute one 
    time only.
    """
    self.initialized = True
    if self.ids_to_consider is None:
      self.V_tot = np.sum(self.V[self.p_ids-1])
    else:
      self.V_tot = np.sum(self.V[self.ids_to_consider-1])
    return
  
  
  ### REQUIRED FOR CRAFTING ###
  def __require_adjoint__(self): return False
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed
  def __get_name__(self): return "Volume"
  def __get_constrain_tol__(self): return self.max_v_frac

