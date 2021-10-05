import time
from scipy.sparse import coo_matrix, dia_matrix
import Adjoint_Solve
import numpy as np


class Sensitivity_Richards:
  """
  Compute the derivative of the cost function according to the material
  distribution parameter p in Richards mode.
  Arguments:
  Note: vector derivative should be numpy array, and matrix in (I,J,data) 
  format as output by PFLOTRAN.get_sensitivity() method.
  If cost_deriv_mat_prop is None, assume the cost function does not depend on
  the material property distribution.
  """
  def __init__(self, parametrized_mat_props, solver, p_ids):
    
    #vector
    #self.dc_dP = cost_deriv_pressure #dim = [cost] * L * T2 / M
    #self.dc_dXi = cost_deriv_mat_prop #[cost] / [mat_prop]
    self.parametrized_mat_props = parametrized_mat_props
    self.solver = solver
    self.assign_at_ids = p_ids #in PFLOTRAN format!
    
    self.adjoint = Adjoint_Solve.Adjoint_Solve()
    
    self.dXi_dp = None
    self.dR_dXi = None
    self.dR_dP = None
    self.initialized = False
    return
    
  def set_adjoint_problem_algo(self, algo=None):
    if algo is not None: self.adjoint.method = algo
    return
    
  def update_mat_derivative(self, p):
    for i,mat_prop in enumerate(self.parametrized_mat_props):
      mat_prop.d_mat_properties(p, self.dXi_dp[i])
    return
  
  def update_residual_derivatives(self):
    self.solver.get_sensitivity("LIQUID_PRESSURE", coo_mat=self.dR_dP)
    for i,mat_prop in enumerate(self.parametrized_mat_props):
      self.solver.get_sensitivity(mat_prop.get_name(), coo_mat=self.dR_dXi[i])
    return 
  
  
  def compute_sensitivity(self, p, dc_dP, dc_dXi, Xi_name):
    """
    Compute the total cost function derivative according to material density
    parameter p.
    Argument:
    - p : the material parameter
    - dc_dP : derivative of the function wrt pressure (PFLOTRAN ordering)
    - dc_dXi : derivative of the function wrt function inputs (p ordering)
    - Xi_name : name of the function input variables
    """
    #create or update structures
    if self.initialized == False:
      self.__initialize_adjoint__(p)
    else:
      self.update_mat_derivative(p)
      self.update_residual_derivatives()
    
    #compute adjoint
    l = self.adjoint.solve(self.dR_dP, dc_dP) #PFLOTRAN ordering
    
    #compute dc/dp
    #note: dR_dXi in PFLOTRAN ordering, so we convert it to p ordering with assign_at_ids
    # and dXi_dP in p ordering
    #thus: dR_dXi_dXi_dp in p ordering
    temp = coo_matrix( ( self.dXi_dp[0], 
                  (np.arange(len(self.dXi_dp[0])),np.arange(len(self.dXi_dp[0])) ) ) 
                     )
    dR_dXi_dXi_dp =  ((self.dR_dXi[0]).tocsr())[:,self.assign_at_ids-1] * temp.tocsr()
    if self.n_parametrized_props > 1:
      for i in range(1,self.n_parametrized_props):
        temp.data = self.dXi_dp[i]
        dR_dXi_dXi_dp += \
            ((self.dR_dXi[i]).tocsr())[:,self.assign_at_ids-1] * temp
    
    dc_dXi_dXi_dp = 0.
    for i,mat_prop in enumerate(self.parametrized_mat_props):
      for j,name in enumerate(Xi_name):
        if name == mat_prop.name:
          dc_dXi_dXi_dp += dc_dXi[j][self.assign_at_ids-1]*self.dXi_dp[i]
    #if self.assign_at_ids is not None and isinstance(dc_dXi_dXi_dp,np.ndarray):
    #  dc_dXi_dXi_dp = dc_dXi_dXi_dp[self.assign_at_ids-1]
      
    S = dc_dXi_dXi_dp - (dR_dXi_dXi_dp.transpose()).dot(l)
    
    return S
  
  
  def __initialize_adjoint__(self,p): 
    self.dXi_dp = [mat_prop.d_mat_properties(p) for mat_prop in self.parametrized_mat_props] 
               # dim = [mat_prop] * L * T2 / M
    self.n_parametrized_props = len(self.dXi_dp)
      
    self.dR_dXi = [self.solver.get_sensitivity(mat_prop.get_name()) 
                                            for mat_prop in self.parametrized_mat_props]
                      
    self.dR_dP = self.solver.get_sensitivity("LIQUID_PRESSURE")
    
    self.initialized = True
    return
    

