[![Static Badge](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.2c00674-blue?style=flat&logo=DOI)
](https://doi.org/10.1021/acs.jctc.2c00674)

# EDUS - Electron Dynamics and Ultrafast Spectroscopy

This code can be used to propagate the electronic density matrix of an extended system up to HSEX level. It can simulate the interaction of the system with a (classical) electric field. \
The equations we solve are:<br />
   $`i{{\partial \rho(\textbf{k})}\over{\partial t}} = \Big[ H_0(k) +E(t)\cdot \xi(k) + H_{ee}(k), \rho(k)\Big] + i E(t)\cdot \nabla_k \rho(k)`$

The theory behind the code plus details about the implementation can be found in our paper [Theoretical Approach for Electron Dynamics and Ultrafast Spectroscopy (EDUS)](https://doi.org/10.1021/acs.jctc.2c00674). If you find our paper or the code useful, please consider citing us.

---

## Building instructions
#### Requirements
Basics:
```bash
gcc13
cmake
git
```
#### Link dependencies
    • MKL (for lapack/blas routines, eventually for fftw)
    • MPI (openmpi/mpich)
    • fftw (mpi version if you want to compile with mpi)
    • hdf5 (for outputing results in an output.h5 data file) 
    • python3 + numpy (at runtime, for building the RytovaKeldysh potential)

It is important to note that the `mpi`, `fftw`, and `hdf5` must have matching version numbers. I.e.,
```bash
gcc/13.2.0
openmpi_gcc/5.0.2_gcc13.2.0
fftw/3.3.10_mpi5_gcc13
hdf5/1.14.3_gcc13_mpi5
```

Additionally, the load order is important for compilation. Ideally, the modules should be loaded as  
```bash
module purge
module load python/3.11.4 gcc/13.2.0 intel/2022.3 openmpi_gcc/5.0.2_gcc13.2.0 fftw/3.3.10_mpi5_gcc13 hdf5/1.14.3_gcc13_mpi5
```


#### Building
For all distros:
```bash
git clone https://github.com/gcistaro/EDUS_2.0.git
cd EDUS_2.0
mkdir build; cd build
cmake ..
make
ctest
```
#### Building with spack
To install EDUS with SPACK, first of all you need a version of spack in your computer
```bash
git clone -c feature.manyFiles=true https://github.com/spack/spack.git ~/spack
cd spack
git checkout v0.23.1
```
And need to source the setup file in spack: 
```bash
. ~/spack/share/spack/setup-env.sh
```
Now spack is available. Try to find some packages that are usually already installed
```bash
spack external find autoconf automake git libtool m4 perl tar
spack compiler add
```
Now that you have spack, you can create a new environment. For example, create a folder:
```bash
cd
mkdir envs
cd envs
mkdir EDUS
cd EDUS
```
This will be the folder with your environment. Inside it, create a file called spack.yaml and inside copy this:
```bash
# This is a Spack Environment file.
#
# It describes a set of packages to be installed, along with
# configuration settings.
spack:
  # add package specs to the `specs` list
  specs:
  - EDUS@1.0%gcc@12.3.0 build_type=Debug
  view: true
  concretizer:
    unify: true
  develop:
    EDUS:
      path: /users/gcistaro/codes/EDUS_2.0
      spec: EDUS@1.0
  repos:
  - /users/gcistaro/codes/EDUS_2.0/spack/
```
Be careful to change your path with the one of the code. 
Activate the environment you have just created and concretize it:
```bash
spacktivate . -p
spack concretize 
```

After this is done, you can install.

```bash
spack install -v 
```

---

## Input file
Sample input files can be found in `<repository>/ci-test/inputs`. For now you need to define all the variables, we will soon assign default values to many of the variables. 

Currently, the code only supports reading the coulomb interaction from a file. This file is created automatically when the computation starts by calling the script `PostProces/RytovaKeldysh.py` using the screening length $`r_0`$ (in Angstrom) and the dielectric constant of the surrounding medium $`\epsilon`$ (dimensionless) as defined in the input file. 

[comment]: # (by calling it as `PostProces/RytovaKeldysh.py <nk1> <nk2> <nk3> file_tb.dat`, where `<nk1> <nk2> <nk3>` are the number of kpoints in each cartesian direction and `file_tb.dat` is the Wannier90 output that will be used in the computation. )

## Add new variable to the json file
First, modify src/InputVariables/input_schema.json. 
You need to specify the name of the variable, the variable type under the section "type" (ex. "string", "number", "array").
We also suggest to put a default value for it that the code will take if the input parameter is not specified in the input, and a description under the section "title".
After this, you can modify the config.hpp class adding your new input variable. you need to add two more methods in this format:
```///<title>
    inline auto <parameter-name>() const
    {
        return dict_.at("/<parameter-name>"_json_pointer).get<<parameter-type>>();
    }
    inline void <parameter-name>(<parameter-type> <parameter-name>__)
    {
        if (dict_.contains("locked")) {
            throw std::runtime_error(locked_msg);
        }
        dict_["/<parameter-name>"_json_pointer] = <parameter-name>__;
    }
```
Now that you defined it, make sure the parameter is used in the Simulation class to do what it is supposed to do.
(if not, it will just be read and ignored).
It would be nice that you also print it in print_recap in Simulation.cpp to keep track of it.

---

## Usage

Upon building, the EDUS executable will be inside the build directory. For a generic `input.json.in` file, EDUS can be run from the root of the project as
```
./build/EDUS ./path/to/input.json.in 1> output.log 2> output.err
```

---
### Print band dispersion
EDUS can be used to print band dispersions when a simulation is performed. To do that, you need to define an array of arrays 
in the input json like this one: 
```
    "kpath": [ [0.000, 0.000, 0.000],
               [0.500, 0.500, 0.000],
               [0.333, 0.667, 0.000],
               [0.000, 0.000, 0.000] ]

```
N.B.: only crystal coordinates are supported for now. This will create a file `BANDSTRUCTURE.txt` and a file to plot it
`plotbands.gnu`. To get a visualization of the bandstructure, just type in the terminal:
```
gnuplot plotbands.gnu
```

### Open gap using scissor operator
You can open the gap using a simple scissor operator that does not change the eigenvectors, but only pushes the bands away from each others using the desired gap. The way to do it is to simply add a parameter in the json file: 
```
"gap" : 10.0,
```
this line pushes the bands until they are not at 10 eV of distance. 
---

## Adding new code
1. Before adding something new, make sure you are not repeating something is already there.
2. To add new code, create a new class, unless you don’t just want to add some features (methods) in a class that already exists. Whenever you want to push something in the remote folder, do not do it in the main branch, but create a new branch with small changes, only related to what you want to add. When you are done with the changes, the first step is to make sure that you did not break old parts of the code, unless you did not find a bug. So, make everything and run ctest. If your tests are failing, consider what you changed in the “old” code that was working before. 
3. To create a new branch, give meaningful names, so that if it stays there for years we will remember what we were doing in that branch.
4. IMPORTANT:  push only the files you want to really to change. To see what you changed, you can use the terminal command git status. If something changed and you did not change it, you can run `git diff <file>`, that will compare your local file with the remote one. If you did some changes (even spaces) that are not needed, you can restore the remote file with `git restore <file>`.
5. After your changes are done, create a pull request and we will add them in main.

### Adding a class
If you want to add a class from scratch, you need to add to the code two different files:
1. A header file .hpp, that you will link to all the parts of the code that need your class using `#include <”file”.hpp>`. This class contains the declaration of your class together with the declaration of all the methods. “Declaration” means a line where you define return type, name, arguments but never their definition. This will be included in many parts of the code, so make sure it will be compiled only once with #ifdef.
2. A source file .cpp, which contains the actual code to be used. You need to add this file in the list of __SOURCES in CMakeLists.txt so it will be compiled as an object file with the code. In this file you need to write the definition of everything you just declared in your .hpp file. 

