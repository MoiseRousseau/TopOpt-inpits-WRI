import time
import scipy.sparse.linalg as spla
from scipy.sparse import dia_matrix
import scipy.sparse as sp
import numpy as np



class Adjoint_Solve:
  def __init__(self, algo=""):
    self.algo = algo
    self.last_l = None
    if "gpu" in self.algo: 
      print("WARNING: gpu solve is highly experimental")
    
    #default parameter for some algorithm
    self.cg_tol = 5e-4
    self.cg_preconditionner = "jacobi"
    return
  
  def solve(self, A, b):
    #default parameter
    if self.algo == "":
      if len(b) > 60000: self.algo = "bicgstab"
      else: self.algo = "lu"
    #solve
    print(f"Solve adjoint equation using {self.algo}")
    t_start = time.time()
    if self.algo == "spsolve":
      l = self.__sp_solve__(A,b)
    if self.algo == "lu":
      l = self.__lu_solve__(A,b)
    elif self.algo == "bicgstab":
      l = self.__bicgstab_solve__(A,b)
    elif self.algo == "cg_gpu":
      l = self.__cg_solve_gpu__(A,b)
    print(f"Time to solve adjoint: {(time.time() - t_start)} s")
    return l
  
  def __sp_solve__(self, A, b):
    _A = A.tocsr()
    l = spla.spsolve(_A, b) 
    return
  
  def __lu_solve__(self, A, b):
    _A = A.tocsc() #[L-1]
    LU = spla.splu(_A)
    l = LU.solve(b)
    return l
  
  def __bicgstab_solve__(self, A, b):
    #always use jacobi preconditioning
    D_ = dia_matrix((np.sqrt(1/A.diagonal()),[0]), shape=A.shape)
    _A = D_ * A * D_
    _b = D_ * b
    l, info = spla.bicgstab(_A, _b, x0=self.last_l, 
                              tol=self.cg_tol, atol=-1) #do not rely on atol
    #copy for making starting guess for future iteration
    if self.last_l is None: self.last_l = np.copy(l)
    else: self.last_l[:] = l
    if info: 
      print("Some error append during BiConjugate Gradient Stabilized solve")
      print(f"Error code: {info}")
      exit(1)
    l = D_ * l
    return l
  
  def __cg_solve_gpu__(self, A,b):
    #always use jacobi preconditioning
    D_ = dia_matrix((np.sqrt(1/A.diagonal()),[0]), shape=A.shape)
    #set up gpu array
    _A = css.csr_matrix(D_ * A * D_)
    _b = cupy.array(D_ * b)
    #solve gpu
    l, info = css.linalg.cg(_A,_b, x0=self.last_l, tol=self.cg_tol, atol=-1)
    if self.last_l is None: self.last_l = cupy.copy(l)
    else: self.last_l[:] = l
    if info: 
      print("Some error append during Conjugate Gradient GPU solve")
      print(f"Error code: {info}")
      exit(1)
    return np.array(D_ * l)
    


