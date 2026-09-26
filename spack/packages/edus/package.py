from spack.package import *


class Edus(CMakePackage):
    """Tool to propagate the Density Matrix of electrons in time in the HSEX approximation."""

    homepage = "https://github.com/gcistaro/EDUS_2.0" 
    url = "https://github.com/gcistaro/EDUS_2.0/archive/main.tar.gz"

    maintainers("gcistaro")


    version("1.0", sha256="018827201601495e27cbd4e5434cd0b51e635099")

    depends_on("cxx", type="build")
    depends_on("c", type="build")

    depends_on("cmake", type="build")
    depends_on("intel-oneapi-mkl", type="build")
    depends_on("fftw+mpi+openmp", type="build")
    depends_on("hdf5", type="build")
    depends_on("pkgconfig", type="build")
    depends_on("mpi")

    variant(
        "build_type",
        default="Release",
        description="CMake build type",
        values=("Debug", "Release", "RelWithDebInfo"),
    )
    #def autoreconf(self, spec, prefix):
    #    # FIXME: Modify the autoreconf method as necessary
    #    autoreconf("--install", "--verbose", "--force")

    def cmake_args(self):
        args = []
        #    "-DWHATEVER:STRING=somevalue",
        #    self.define("ENABLE_BROKEN_FEATURE", False),
        #    self.define_from_variant("DETECT_HDF5", "hdf5"),
        #    self.define_from_variant("THREADS"), # True if +threads
        #]

        return args
