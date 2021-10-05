import Mesh_NNR
import numpy as np
from scipy.sparse import dia_matrix


class Density_Filter:
  """
  Filter the density parameter according to Bruns and Tortorelli (2001) and 
  Bourdin (2001):
  https://doi.org/10.1016%2FS0045-7825%2800%2900278-4
  https://doi.org/10.1002%2Fnme.116
  Summarized in:
  https://link.springer.com/article/10.1007/s00158-009-0452-7
  
  Argument:
  - filter_radius: scalar or list of scalar specifying the filter radius in X,Y,Z direction
  - distance_weighting_power: distance weighting function power
                              positive mean higher weigth (p value more correlated)
  """
  def __init__(self, filter_radius=1., distance_weighting_power=1):
    self.filter_radius = filter_radius
    if distance_weighting_power <= 0.: 
      print("distance_weighting_power argument must be strictly positive")
      raise ValueError
    self.distance_weighting_power = distance_weighting_power
    self.p_ids = None
    self.volume = None
    self.mesh_center = None
    self.neighbors = None
    self.initialized = False
    
    self.output_variable_needed = ["X_COORDINATE", "Y_COORDINATE",
                                   "Z_COORDINATE", "VOLUME"]
    return
  
  def set_p_to_cell_ids(self, p_ids):
    self.p_ids = p_ids #if None, this mean all the cell are parametrized
    return
  
  def set_inputs(self, inputs):
    self.volume = inputs[3]
    self.mesh_center = np.array(inputs[:3]).transpose()
    return
  
  
  def initialize(self):
    if self.p_ids is not None:
      V = self.volume[self.p_ids-1] #just need those in the optimized domain
      X = self.mesh_center[self.p_ids-1,:]
    else:
      V = self.volume
      X = self.mesh_center
    if isinstance(self.filter_radius, list): #anisotropic
      for i in range(3): X[:,i] /= self.filter_radius[i]
      R = 1.
    else:
      R = self.filter_radius
    print("Build kDTree and compute mesh fixed radius neighbors")
    self.neighbors = Mesh_NNR.Mesh_NNR(X)
    self.neighbors.find_neighbors_within_radius(R)
    self.D_matrix = -self.neighbors.get_distance_matrix().tocsr(copy=True)
    self.D_matrix.data += R
    self.D_matrix.data = self.D_matrix.data ** self.distance_weighting_power
    self.D_matrix = self.D_matrix.dot( dia_matrix((V[np.newaxis,:],0),
                                                   shape=self.D_matrix.shape) )
    self.deriv = self.D_matrix.multiply(1/self.D_matrix.sum(axis=1)).transpose()
    self.initialized = True
    return
  
  def get_filtered_density(self, p, p_bar=None):
    if not self.initialized: self.initialize()
    if p_bar is None:
      p_bar = np.zeros(len(p), dtype='f8')
    temp = self.D_matrix.dot( dia_matrix((p[np.newaxis,:],0),shape=self.D_matrix.shape) )
    p_bar[:] = (temp.sum(axis=1) / self.D_matrix.sum(axis=1)).flatten()
    return p_bar
  
  def get_filter_derivative(self, p):
    if not self.initialized: self.initialize()
    out = self.deriv
    return out
 
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  

