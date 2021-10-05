# TopOpt-inpits-WRI
Topology optimization of in-pits wastes rocks inclusions to limit groundwater flowrate in contaminated tailings

## Getting started

This repository contains two examples of topology optimization of the codisposition of waste rock and tailings in backfilled pit to minimise groundwater flow in tailings.
The two examples are located in the `examples` folder and the optimization scripts in the `src` folder.

Note all the scripts were developed and tested for Linux Ubuntu. They might very probably work without modifications on other Linux distro (MacOS also?), but getting them work on Windows might need further developement and verification.
However, for Windows user, you can still install [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install) or using a [Virtual Machine](https://www.virtualbox.org/wiki/Downloads).

Before running the proposed example, you will need to install the developement version of PFLOTRAN containing the process model performing the sensitivity analysis.
Thus, follow first the PFLOTRAN installation instructions for your machine available [here](http://doc-dev.pflotran.org/user_guide/how_to/installation/installation.html).
Then, when compiling PFLOTRAN (step 5 in linux installation instructions), checkout the right developement branch with:
```
cd pflotran/src/pflotran
git checkout moise/make_optimization_v2
make pflotran
```

At this stage, PFLOTRAN is installed.
If you never use PFLOTRAN before, you may want to have quick look at the [User Guide](http://doc-dev.pflotran.org/user_guide/user_guide.html) to know more about PFLOTRAN.

To run the optimization, you also need several Python libraries which can be install with:
```
pip3 install numpy scipy nlopt h5py
```


## Run the examples

### Instruction

There is 2 examples in this repository, a light-weight two-dimensional optimization problem running in less than 5 minutes, and a larger three-dimensional problem which can be run within an hour.

For both example, the main optimization script is the `make_optimization.py` file which can be run by opening a terminal (on Linux, right click/open a terminal) and run `python make_optimization.py`.

What does the main script do ?
1. imports the necessary libraries and other scripts to perform the optimization.
2. reads the PFLOTRAN input file on which the optimization is perform
3. parametrizes the material hydraulic conductivity in the pit
4. defines the cost function (mean head in the tailings) and define the volume constrain
5. sets the density filter
6. crafts the optimization problem
7. initializes the GC-MMA optimize
8. performs the optimization


### Two-dimensional pit

The 2D example consists in a pit backfilled with contaminated mine talings of 5e-7 m/s hydraulic conductivity.
The objective is to use highly permeable waste rock inclusion of 5e-4 m/s hydraulic conductivity to create preferential patch in the pit to reduce the mean head gradient in the tailings and thus the groundwater flow through them.
In this example, wastes rock inclusion  that can occupy 20% of the pit volume.
Regional head gradient is 0.01 along the x direction.
The following example showed the evolution of the geometry of the inclusion (in red) within the contaminated tailings (in blue) from the start of the iterative optimization process until the end.
The mean head gradient in the tailings is being reduced from 3.6e-3 to 2e-5, i.e. a two order of magnitude reduction.

![2D example](https://github.com/MoiseRousseau/TopOpt-inpits-WRI/blob/main/examples/2D_pit/results.gif)


### Three-dimensional pit

The 3D example consists in a backfilled pit with the same settings as the 2D example above.
In this case, the backfilled pit is discretized using 48000 Voronoi cells and the iterative optimization process starts from a random discrete distribution and give a pervious surround at the end.
Mean head gradient is reduced from 4.7e-3 without inclusion to 8.2e-5, i.e. a 60 times reduction.

![3D example](https://github.com/MoiseRousseau/TopOpt-inpits-WRI/blob/main/examples/3D_pit/results.gif)


## Further help

The discussion section (see tabs above) is the right place to ask question.
You can also reached me by mail at: rousseau \<dot\> moise \<at\> gmail \<dot\> com.

This proto-library is still under development, stay in tune for further update!
