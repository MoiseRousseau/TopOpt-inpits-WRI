import numpy as np
from scipy.sparse import coo_matrix, dia_matrix
import h5py
import subprocess
import os
import time

default_gravity = 9.8068
default_viscosity = 8.904156e-4
default_water_density = 997.16

class PFLOTRAN:
  """
  This class make the interface between PFLOTRAN and the calculation
  of sensitivity derivative and the input
  """
  def __init__(self, pft_in, mesh_info=None):
    #input related
    self.pft_in = pft_in
    self.input_folder = '/'.join(pft_in.split('/')[:-1])+'/'
    self.prefix = (self.pft_in.split('/')[-1]).split('.')[0]
    self.output_sensitivity_format = "HDF5" #default
    if self.input_folder[0] == '/': self.input_folder = '.' + self.input_folder
    self.__get_input_deck__(self.pft_in)
    self.mesh_type = None
    self.__get_mesh_info__()
    self.__get_nvertices_ncells__()
    self.__get_sensitivity_info__()
    
    #running
    self.mpicommand = ""
    self.nproc = 1
    self.no_run = False #boolean flag to not run PFLOTRAN for debugging
    
    #output
    self.pft_out = '.'.join(pft_in.split('.')[:-1])+'.h5'
    self.pft_out_sensitivity = '.'.join(pft_in.split('.')[:-1]) + "-sensitivity-flow"
    if self.mesh_type in ["ugi", "h5"]:
      self.domain_file = self.pft_out
    else:
      self.domain_file = self.__get_domain_filename__()
    if mesh_info is None:
      self.mesh_info = self.pft_out
      self.mesh_info_present = False
    else:
      self.mesh_info = mesh_info
      self.mesh_info_present = True
    
    #for internal working
    self.dict_var_out = {"FACE_AREA" : "Face Area", 
                         "FACE_DISTANCE_BETWEEN_CENTER" : "Face Distance Between Center",
                         "FACE_UPWIND_FRACTION" : "Face Upwind Fraction",
                         "FACE_NORMAL_X": "Face Normal X Component",
                         "FACE_NORMAL_Y": "Face Normal Y Component",
                         "FACE_NORMAL_Z": "Face Normal Z Component",
                         "FACE_CELL_CENTER_VECTOR_X": "Face Cell Vector X Component",
                         "FACE_CELL_CENTER_VECTOR_Y": "Face Cell Vector Y Component",
                         "FACE_CELL_CENTER_VECTOR_Z": "Face Cell Vector Z Component",
                         "LIQUID_CONDUCTIVITY" : "Liquid Conductivity",
                         "LIQUID_PRESSURE" : "Liquid Pressure",
                         "PERMEABILITY" : "Permeability",
                         "VOLUME" : "Volume", 
                         "X_COORDINATE" : "X Coordinate",
                         "Y_COORDINATE" : "Y Coordinate",
                         "Z_COORDINATE" : "Z Coordinate"}
    self.dict_var_sensitivity_matlab = \
         {"PERMEABILITY":"permeability","LIQUID_PRESSURE":"pressure"}
    self.dict_var_sensitivity_hdf5 = \
         {"PERMEABILITY":"Permeability []","LIQUID_PRESSURE":"Pressure []"}
    return
    
  def set_parallel_calling_command(self, processes, command):
    """
    Specify the command line argument for running PFLOTRAN related to 
    parallelization.
    Arguments:
    - processes: the number of core to run PFLOTRAN (the -n argument for mpirun)
    - command: the MPI command (example: mpiexec.mpich)
    """
    self.mpicommand = command
    self.nproc = processes
    return
    
  def get_grid_size(self):
    return self.n_cells
  
  def disable_run(self):
    self.no_run = True
    return
  
  
  # interacting with data #
  def get_region_ids(self, reg_name):
    """
    Return the cell ids associated to the given region:
    - reg_name: the name of the region to get the ids.
    """
    #look for region in pflotran input
    filename = ""
    for i,line in enumerate(self.input_deck):
      if "REGION" in line and reg_name in line:
        line = self.input_deck[i+1]
        if "FILE" in line:
          line = line.split()
          index = line.index("FILE")
          filename = self.input_folder+line[index+1]
          break
    if not filename:
      print(f"No region \"{reg_name}\" found in PFLOTRAN input file, stop...")
      exit(1)
    
    #try hdf5
    try:
      src = h5py.File(filename, 'r')
      if reg_name in src["Regions"]:
        cell_ids = np.array(src["Regions/"+reg_name+"/Cell Ids"])
        src.close()
      else:
        src.close()
        print(f"Region not found in mesh file {filename}")
        exit(1)
    #else ascii
    except:
      try:
        cell_ids = np.genfromtxt(filename, dtype='i8')
      except:
        print(f"File {filename} not readable")
    return cell_ids 
  
  def get_connections_ids_integral_flux(self, integral_flux_name):
    """
    Return the cell ids associated to the given integral flux
    Argument:
    - integral_flux_name: the name of the integral flux to get the ids.
    """
    found = False
    #find the integral flux in input deck
    for i,line in enumerate(self.input_deck):
      if "INTEGRAL_FLUX" in line:
        if integral_flux_name in line: 
          found = True
          break
    if not found:
      print(f"Integral flux {integral_flux_name} not found in input deck")
      print("Please provide the name directly after the INTEGRAL_FLUX opening card")
      exit(1)
    #check if defined by cell ids
    while "CELL_IDS" not in line:
      i += 1
      line = self.input_deck[i]
      for x in ["POLYGON", "COORDINATES_AND_DIRECTIONS", 
                "PLANE", "VERTICES", "END", "/"]:
        if x in line:
          print("Only INTEGRAL_FLUX defined with CELL_IDS are supported at this time")
          exit(1)
    cell_ids = []
    i += 1
    line = self.input_deck[i].split()
    while "/" not in line and "END" not in line:
      cell_ids.append([int(line[0]),int(line[1])])
      i += 1
      line = self.input_deck[i].split()
    if len(cell_ids) == 0:
      print(f"No connections found in the INTEGRAL_FLUX card \"{integral_flux_name}\"")
      exit(1)
    return np.array(cell_ids)
  
  def create_cell_indexed_dataset(self, X_dataset, dataset_name, h5_file_name="",
                                  X_ids=None, resize_to=True):
    """
    Create a PFLOTRAN cell indexed dataset.
    Arguments:
    - X_dataset: the dataset
    - dataset_name: its name (need to be the same as in PFLOTRAN input deck)
    - h5_file_name: the name of the h5 output file (same as in PFLOTRAN input deck)
    - X_ids: the cell ids matching the dataset value in X
             (i.e. if X_ids = [5, 3] and X_dataset = [1e7, 1e8],
             therefore, cell id 5 will have a X of 1e7 and 3 with 1e8).
             By default, assumed in natural order
    - resize_to: boolean for resizing the given dataset to number of cell
                 (default = True)
    """
    #first cell is at i = 0
    if not h5_file_name: h5_file_name=dataset_name.lower()+'.h5'
    h5_file_name = self.input_folder + h5_file_name
    out = h5py.File(h5_file_name, 'w')
    if X_ids is None: resize_to=False
    if resize_to and self.n_cells != len(X_dataset):
      if X_ids is None:
        print("Error: user must provide the cell ids corresponding to the dataset since the length of the dataset length does not match the number of cell in the grid")
        exit(1)
      X_new = np.zeros(self.n_cells, dtype='f8')
      X_new[X_ids.astype('i8')-1] = X_dataset
      X_dataset = X_new
      X_ids = None
    out.create_dataset(dataset_name, data=X_dataset)
    if X_ids is None:
      out.create_dataset("Cell Ids", data=np.arange(1,len(X_dataset)+1))
    else:
      out.create_dataset("Cell Ids", data=X_ids)
    out.close()
    return
  
  
  def write_output_variable(self, X_dataset, dataset_name, h5_file_name, h5_mode,
                                  X_ids=None):
    """
    Write the "X_dataset" in the HDF5 file "h5_file_name" and link it to the input mesh
    for visualization
    """
    #correct X_dataset if not of the size of the mesh
    if len(X_dataset) != self.n_cells:
      if X_ids is None: 
        print("Error: user must provide the cell ids corresponding to the dataset")
        return 1
      X_new = np.zeros(self.n_cells, dtype='f8')-999
      X_new[:] = np.nan
      X_new[X_ids-1] = X_dataset
      X_dataset = X_new
    #write it
    out = h5py.File(h5_file_name, h5_mode)
    out.create_dataset(dataset_name, data=X_dataset)
    out.close()
    return
  
  def write_output_xmdf(self, out_number, out_file, var_list, var_name):
    """
    Write a xdmf file for visualizing the variable "in var_list"
    """
    if out_number < 10: out_number = "00"+str(out_number)
    elif out_number < 100: out_number = "0"+str(out_number)
    out_xdmf = '.'.join(out_file.split(".")[:-1]) + f"-{out_number}.xmf"
    out = open(out_xdmf, 'w')
    self.__write_xdmf_header__(out,out_number)
    self.__write_xdmf_grid__(out)
    for att_name, att_var in zip(var_name,var_list):
      self.__write_xdmf_attribute__(out, out_file, att_name, att_var)
    self.__write_xdmf_footer__(out)
    out.close()
    return out_xdmf

  
  # running
  def run_PFLOTRAN(self):
    """
    Run PFLOTRAN. No argument method
    """
    if self.no_run: return 0
    print("Running PFLOTRAN: ",end='')
    if self.mpicommand:
      cmd = [self.mpicommand, "-n", str(self.nproc), "pflotran", "-pflotranin", self.pft_in]
    else:
      cmd = ["pflotran", "-pflotranin", self.pft_in]
    tstart = time.time()
    ret = subprocess.call(cmd, stdout=open("PFLOTRAN_simulation.log",'w'))
    print(f"{time.time() - tstart} s to run simulation")
    if ret: 
      print("\n!!! Error occured in PFLOTRAN simulation !!!")
      print(f"Please see {self.input_folder}PFLOTRAN_simulation.log for more details\n")
    return ret
  
  
  ### INTERACT WITH OUTPUT DATA ###
  def initiate_output_cell_variable(self):
    return np.zeros(self.n_cells, dtype='f8')
  
  
  def get_internal_connections(self, out=None):
    """
    Return the internal connection of the mesh
    """
    src = h5py.File(self.mesh_info, 'r')
    if "Domain" in list(src.keys()): prefix = "Domain/"
    else: prefix = ""
    try:
      out = np.array(src[prefix+"Connection Ids"])
    except:
      print(f"\nOutput variable \"Domain/Connection Ids\" not found in PFLOTRAN output")
      print(f"Do you forgot to add the \"PRINT_CONNECTION_IDS\" output variable under \
              the OUTPUT,SNAPSHOT_FILE card?\n")
      exit(1)
    src.close()
    return out
  
  
  def get_output_variable(self, var, out=None, i_timestep=-1):
    """
    Return output variable after simulation
    If out array is provided, copy variable to array, else, create a new one
    Arguments:
    - var: the variable name as in PFLOTRAN input file under VARIABLES block
    - out: the numpy output array (default=None)
    - timestep: the i-th timestep to extract
    """
    #treat coordinate separately as they are in Domain/XC unless for uge grid
    if var in ["X_COORDINATE", "Y_COORDINATE", "Z_COORDINATE"] and \
                                            self.mesh_type not in ["uge","h5e"]:
      var = var[0]+"C"
      src = h5py.File(self.mesh_info, 'r')
      if out is None:
        out = np.array(src["Domain/"+var])
      else:
        out[:] = np.array(src["Domain/"+var])
      src.close()
      return out
    #treat separately face normal as it is not ouputted by PFLOTRAN
#    if var in ["FACE_NORMAL_X", "FACE_NORMAL_Y", "FACE_NORMAL_Z"] and \
#      self.mesh_type == "h5e":
#      src = h5py.File(self.input_folder + self.mesh_file, 'r')
#      if "Normals" in src["Domain/Connections"]:
#        temp = np.array(src["Domain/Connections/Normals"])
#        if var == "FACE_NORMAL_X": temp = temp[:,0]
#        elif var == "FACE_NORMAL_Y": temp = temp[:,1]
#        else: temp = temp[:,2]
#        if out is None:
#          out = temp
#        else:
#          out[:] = temp
#        print(var, temp[:100])
#        return out
#      else:
#        print(f"{var} information not available, switch to FACE_CELL_CENTER_VECTOR instead")
#        var = "FACE_CELL_CENTER_VECTOR_" + var[-1]
#      src.close()
    if var in ["FACE_NORMAL_X", "FACE_NORMAL_Y", "FACE_NORMAL_Z"] and \
              self.mesh_type in ["uge","h5e"]:
      print(f"{var} information not available, switch to FACE_CELL_CENTER_VECTOR instead")
      var = "FACE_CELL_CENTER_VECTOR_" + var[-1]
    #treat separately grid output since they could be in the mesh_info file
    if var in ["LIQUID_CONDUCTIVITY","LIQUID_PRESSURE","PERMEABILITY"]:
      f_src = self.pft_out
    else:
      f_src = self.mesh_info
    #other variable
    src = h5py.File(f_src, 'r')
    timesteps = [x for x in src.keys() if "Time" in x]
    right_time = timesteps[i_timestep]
    key_to_find = self.dict_var_out[var]
    found = False
    for out_var in src[right_time].keys():
      if key_to_find in out_var: 
        found = True
        break
    if not found:
      print(f"\nOutput variable \"{self.dict_var_out[var]}\" not found in PFLOTRAN output")
      print(f"Available variable are:")
      print(src[right_time].keys())
      print(f"Do you forgot to add the \"{var}\" output variable under the OUTPUT card?\n")
      exit(1)
    if out is None:
      out = np.array(src[right_time + '/' + out_var])
    else:
      out[:] = np.array(src[right_time + '/' + out_var])
    src.close()
    return out
  
  def get_sensitivity(self, var, timestep=None, coo_mat=None):
    # TODO: change the name of the dict_var_sensitivity to match the new output
    """
    Return a (3,n) shaped numpy array (I, J, data) representing the derivative
    of the residual according to the inputed variable. Input variable must be 
    consistent with a material property in the input deck.
    Sensitivity outputed by PFLOTRAN is supposed to be in matlab format
    Arguments:
    - var: the input variable (ex: PERMEABILITY)
    """
    if self.output_sensitivity_format == "HDF5":
       f = self.pft_out_sensitivity + '.h5'
       src = h5py.File(f, 'r')
       i = np.array(src["Mat Structure/Row Indices"])
       j = np.array(src["Mat Structure/Column Indices"])
       if timestep is None: timestep = -1
       list_timestep = list(src.keys())
       temp_str = list_timestep[timestep] + '/' + self.dict_var_sensitivity_hdf5[var]
       data = np.array(src[ temp_str ])
       src.close()
    elif self.output_sensitivity_format == "MATLAB":
      if timestep is None:
        output_file = [x[:-4] for x in os.listdir(self.input_folder) if x[-4:] == '.mat']
        output_file = [x for x in output_file if self.prefix+'-sensitivity-flow-' in x]
        output_file = [int(x.split('-')[-1]) for x in output_file]
        timestep = max(output_file)
      if timestep < 10: timestep = "00"+str(timestep)
      elif timestep < 100: timestep = "0"+str(timestep)
      else: timestep = str(timestep)
      f = self.pft_out_sensitivity + '-' + self.dict_var_sensitivity_matlab[var] \
            + '-' + timestep + '.mat'
      src = np.genfromtxt(f, skip_header=8, skip_footer=2)
      i, j, data = src[:,0], src[:,1], src[:,2]
    if coo_mat is None:
      new_mat = coo_matrix( (data,(i.astype('i8')-1,j.astype('i8')-1)), dtype='f8')
      return new_mat
    else:
      coo_mat.data[:] = data
    return
  
  
  
  
  ### PRIVATE METHOD ###
  def __get_input_deck__(self, filename):
    self.input_deck = []
    self.__read_input_file__(filename)
    finish = False
    while not finish:
      restart = False
      for i,line in enumerate(self.input_deck):
        if "EXTERNAL_FILE" in line:
          self.input_deck.pop(i)
          line_split = line.split()
          index = line_split.index("EXTERNAL_FILE")
          self.__read_input_file__(self.input_folder+line_split[index+1],append_at_pos=i)
          restart = True
          break
      if not restart: finish = True
    return
  
  def __read_input_file__(self, filename, append_at_pos=0):
    """
    Store input deck
    """
    src = open(filename, 'r')
    temp = []
    #read line in source file and remove \n and commentaru
    for line in src.readlines():
      line = line.split('#')[0] 
      if len(line)>0 and line[-1] == '\n': line = line[:-1]
      if line: temp.append(line)
    #remove skip / noskip part
    skip = []
    noskip = [] 
    for i,line in enumerate(temp):
      if "NOSKIP" in line:
        noskip.append(i)
        continue
      if "SKIP" in line: 
        skip.append(i)
    if len(skip) != len(noskip):
      print(skip, noskip)
      print(f"ERROR! number of SKIP does not match the number of NOSKIP in file {filename}")
    for i,j in zip(skip, noskip):
      temp = temp[:i] + temp[j+1:]
    #add the result in the input deck
    if not self.input_deck: self.input_deck = temp
    else:
      self.input_deck = self.input_deck[:append_at_pos] + \
                        temp + self.input_deck[append_at_pos:]
    src.close()
    return 
    
  
  def __get_mesh_info__(self):
    """
    Read PFLOTRAN input deck to get the mesh type and the mesh file
    """
    for line in self.input_deck:
      if "TYPE" and "UNSTRUCTURED_EXPLICIT" in line: 
        if ".h5" in line: self.mesh_type = "h5e"
        else: self.mesh_type = "uge"
        break
      if "TYPE" and "UNSTRUCTURED" in line: 
        if ".h5" in line: self.mesh_type = "h5"
        else: self.mesh_type = "ugi"
        break
      if "TYPE" and "STRUCTURED" in line: 
        self.mesh_type = "struc"
        print("\nWARNING: STRUCTURED grid type not supported\n")
        exit()
        break
    nline = line.split()
    for i,x in enumerate(nline):
      if "STRUCTURED" in x: break
    self.mesh_file = nline[i+1]
    return
  
  def __get_nvertices_ncells__(self):
    """
    Get the number of vertices and cells in the input mesh
    """
    mesh_path = self.input_folder + self.mesh_file
    if self.mesh_type == "h5": #unstructed h5 mesh
      src = h5py.File(mesh_path, 'r')
      self.n_vertices = len(src["Domain/Vertices"])
      self.n_cells = len(src["Domain/Cells"])
      src.close()
      return
    elif self.mesh_type == "ugi": #unstructured ascii mesh
      src = open(mesh_path, 'r')
      line = src.readline()
      self.n_cells, self.n_vertices = [int(x) for x in line.split()]
      src.close()
      return
    elif self.mesh_type == "uge": #unstructured explicit
      src = open(mesh_path, 'r')
      self.n_cells = int(src.readline().split()[1])
      src.close()
      self.n_vertices = -1 #no info about it...
      return
    elif self.mesh_type == "h5e": #unstructured explicit hdf5
      src = h5py.File(mesh_path, 'r')
      self.n_cells = len(src["Domain/Cells/Volumes"])
      src.close()
      self.n_vertices = -1 #no info about it...
      return
  
  def __get_sensitivity_info__(self):
    for i,line in enumerate(self.input_deck):
      if "SENSITIVITY_OUTPUT_FORMAT" in line:
        line = line.split()
        index = line.index("SENSITIVITY_OUTPUT_FORMAT")
        self.output_sensitivity_format = line[index+1].upper()
        break
    return
  
  def __get_domain_filename__(self):
    filename = ""
    for i,line in enumerate(self.input_deck):
      if "DOMAIN_FILENAME" in line:
        line = line.split()
        index = line.index("DOMAIN_FILENAME")
        filename = line[index+1]
        break
    return filename
    
  
  #for xmdf output
  def __write_xdmf_header__(self, out, out_number):
    out.write(f"""\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf>
  <Domain>
    <Grid Name="Mesh">
      <Time Value = "{out_number}" />""")
    return
  def __write_xdmf_footer__(self,out):
    out.write("""
    </Grid>
  </Domain>
</Xdmf>""")
    return
  def __write_xdmf_grid__(self,out):
    dim = -1
    if self.domain_file:
      src = h5py.File(self.domain_file, 'r')
      dim = len(src["Domain/Cells"])
      src.close()
    out.write(f"""
      <Topology Type="Mixed" NumberOfElements="{self.n_cells}">
        <DataItem Format="HDF" DataType="Int" Dimensions="{dim}">
          {self.domain_file}:/Domain/Cells
        </DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem Format="HDF" Dimensions="{self.n_vertices} 3">
          {self.domain_file}:/Domain/Vertices
        </DataItem>
      </Geometry>""")
    return
  def __write_xdmf_attribute__(self, out, out_file, att_name, att_dataset):
    out.write(f"""
      <Attribute Name="{att_name}" AttributeType="Scalar"  Center="Cell">
        <DataItem Dimensions="{self.n_cells} 1" Format="HDF">
          {out_file}:{att_dataset}
        </DataItem>
      </Attribute>""")
    return
    
