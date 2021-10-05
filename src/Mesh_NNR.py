import numpy as np
from scipy.spatial import cKDTree
try:
  from grispy import GriSPy
except:
  pass

class Mesh_NNR:
  """
  Class that compute and store the cell neighbors in a given ball of radius R
  
  Argument:
  - mesh_center: point to compute the fixed radius neighbors
  - method: for development purpose only. Use by default scipy_matrix, which
            is the fastest.
  """
  def __init__(self, mesh_center, method="scipy_matrix"):
    self.mesh_center = mesh_center
    self.method = method
    return
  
  def find_neighbors_within_radius(self, ball_radius):
    #print("Build KDTree")
    if self.method in ["scipy_query", "scipy_matrix"]:
      self.tree = cKDTree(self.mesh_center)
    elif self.method.lower() == "grispy":
      self.tree = GriSPy(self.mesh_center)
    else:
      print("Method for building KDTree not recognized in Mesh_NNR")
      exit(1)
      
    #print("Query neighbors")
    if self.method == "scipy_query":
      self.index = self.tree.query_ball_point(self.mesh_center, ball_radius, n_jobs=-1)
      self.distances = [[np.linalg.norm(self.mesh_center[x[i]] - self.mesh_center[x[0]]) for i in range(1,len(x))] for x in self.index if len(x) > 1]
      #self.mesh_center[self.index[:][0]]# - self.self.mesh_center[self.index[:,1:]]
      #print(self.index, self.distance)
      #process
    if self.method == "scipy_matrix":
      out = self.tree.sparse_distance_matrix(self.tree, ball_radius)
      self.out = out.tocsr() #conversion to csr for fast row indexing for output
    elif self.method == "grispy":
      dist, index = self.tree.bubble_neighbors(self.mesh_center,
                             distance_upper_bound=ball_radius)

    return
  
  def get_neighbors_center(self, cell_id):
    if self.method == "scipy_query":
      index = self.index[cell_id]
      distances = self.distances[cell_id]
    elif self.method == "scipy_matrix":
      temp = self.out.getrow(cell_id)
      index = temp.indices
      distances = temp.data
    return index, distances
    
  def get_distance_matrix(self):
    if self.method == "scipy_matrix":
      return self.out


#test
if __name__ == "__main__":
  import time
  n_pts = 100000
  start = time.time()
  points = np.random.rand(n_pts, 3) * 1000
  FRNN = Mesh_NNR(points, method="scipy_matrix")
  FRNN.find_neighbors_within_radius(20)
  print(FRNN.get_neighbors_center(0))
  print(time.time() - start)
  

