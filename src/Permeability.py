import numpy as np

class Permeability:
  """
  !Link the density parameter to the permeability by a standard \
  SIMP parametrization.
  Input:
  @param bound: permeability bound [minK, maxK]
  @param power: the penalizing power (integer)
  !When p=0 -> k=bound[0]
  !When p=1 -> k=bound[1]
  """
  def __init__(self, bound, cell_ids_to_parametrize=None, power=3, name="PERMEABILITY",
                     reverse=False, log=False):
    if isinstance(cell_ids_to_parametrize, str) and \
             cell_ids_to_parametrize.lower() == "everywhere":
      self.cell_ids = None
    else:
      self.cell_ids = np.array(cell_ids_to_parametrize)
    self.min_K, self.max_K = bound
    self.reverse = reverse
    self.power = power
    self.log = log #log formulation
    self.name= name #note: Permeability class could be used to
                    # parametrize PERMEABILITY_X, PERMEABILITY_Y, ...
    return
  
  def get_cell_ids_to_parametrize(self):
    return self.cell_ids
    
  def convert_p_to_mat_properties(self, p, out=None):
    if out is None: out = np.zeros(len(p),dtype='f8')
    if self.reverse: p_ = 1-p
    else: p_ = p
    if self.log:
      out[:] = 10**(np.log10(self.min_K) + 
                       np.log10(self.max_K/self.min_K)*p_**self.power)
    else:
      out[:] = self.min_K + (self.max_K-self.min_K) * p_**self.power
    return out
  
  def d_mat_properties(self, p, out=None):
    """
    Return the derivative of the material properties according to 
    material parameter p.
    """
    if out is None: out = np.zeros(len(p),dtype='f8')
    if self.reverse: 
      factor = -1.
      p_ = 1-p
    else: 
      factor = 1.
      p_ = p
    if self.log:
      pre = np.log(10) * np.log10(self.max_K/self.min_K) * \
                                  self.convert_p_to_mat_properties(p)
    else:
      pre = (self.max_K-self.min_K)
    out[:] = factor * self.power * pre * p_**(self.power-1)
    return out
      
  def convert_mat_properties_to_p(self, mat_prop_val):
    if np.min(mat_prop_val) >= self.min_K and \
          np.max(mat_prop_val) <= self.max_K :
      p = ( (mat_prop_val - self.min_K) / ( self.max_K - self.min_K) ) ** (1/self.power)
      if self.reverse: p = 1-p
      return p
    else:
      print("Min and max permeability value not in the range of material \
             properties")
      return None
  
  def get_name(self):
    return "PERMEABILITY"
    
  def plot_K_vs_p(self):
    try:
      import matplotlib.pyplot as plt
    except:
      print("Plot requires the matplotlib library")
      return
    p = np.arange(0,101)/100
    K = self.convert_p_to_mat_properties(p)
    dK = self.d_mat_properties(p)
    fig, ax = plt.subplots()
    ax.plot(p,K,'r', label="k")
    ax2 = ax.twinx()
    ax2.plot(p,dK,'b',label="dk/dp")
    ax.set_xlabel("Material parameter p")
    ax.set_yscale("log")
    ax.set_xlim([0,1])
    ax.set_ylabel("Permeability [m^2]")
    ax2.set_ylabel("dPermeability / dp")
    ax.grid()
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)
    plt.show()
    return
    


