#include "GlobalFunctions.hpp"
#include "core/mpi/Communicator.hpp"
#include "initialize.hpp"

std::string LatticeVectors(const Space& space) 
{ 
    return ( space == k ? "k_LatticeVectors" : "R_LatticeVectors" ); 
};


bool
file_exists(std::string file_name)
{
    std::ifstream ifs(file_name.c_str());
    return ifs.is_open();
}


void dump_json_in_h5( const nlohmann::json& data__, const std::string& name__ )
{
#ifdef EDUS_HDF5

#ifdef EDUS_MPI
    if ( mpi::Communicator::world().rank() == 0 ) 
#endif
    {
        HDF5_tree fout(name__, hdf5_access_t::read_write);
        fout.create_node("json");

        fout["json"].write("tb_file", data__["tb_file"].template get<std::string>());
        fout["json"].write("coulomb", data__["coulomb"].template get<int>());
        auto grid = std::vector<int>(3);
        grid = {data__["grid"][0].template get<int>(), 
                data__["grid"][1].template get<int>(), 
                data__["grid"][2].template get<int>()}; 
        fout["json"].write("grid", grid);
        fout["json"].write("filledbands", data__["filledbands"].template get<int>());
        fout["json"].write("dt", data__["dt"].template get<double>());
        fout["json"].write("printresolution", data__["printresolution"].template get<int>());
        fout["json"].write("initialtime", data__["initialtime"].template get<double>());
        fout["json"].write("finaltime", data__["finaltime"].template get<double>());
        fout["json"].write("num_bands", data__["num_bands"].template get<int>());
        fout["json"].write("num_kpoints", data__["num_kpoints"].template get<int>());
        fout["json"].write("solver", data__["solver"].template get<std::string>());
#ifdef EDUS_MPI 
        fout["json"].write("comm_size", kpool_comm->size());
#endif
        for ( int ilaser = 0; ilaser < int( data__["lasers"].size() ); ++ilaser ) {
            auto lasername = "laser"+std::to_string(ilaser);
            fout["json"].create_node(lasername);
            fout["json"][lasername].write("intensity",  data__["lasers"][ilaser]["intensity"].template get<double>());
            fout["json"][lasername].write("frequency",  data__["lasers"][ilaser]["frequency"].template get<double>());
            fout["json"][lasername].write("wavelength", data__["lasers"][ilaser]["wavelength"].template get<double>());
            fout["json"][lasername].write("cycles", data__["lasers"][ilaser]["cycles"].template get<int>());
            auto grid = std::vector<int>(3);
            grid = { data__["lasers"][ilaser]["polarization"][0].template get<int>(), 
                     data__["lasers"][ilaser]["polarization"][1].template get<int>(), 
                     data__["lasers"][ilaser]["polarization"][2].template get<int>()}; 
            fout["json"][lasername].write("polarization", data__["lasers"][ilaser]["cycles"].template get<int>());
        }
        fout["json"].write("kpoints", data__["kpoints"].get<std::vector<double>>());
        fout["json"].write("A", data__["A"].get<std::vector<double>>());
        fout["json"].write("B", data__["B"].get<std::vector<double>>());
    }
#endif
}


namespace output {
    void stars()
    {
#ifdef EDUS_MPI
        if ( mpi::Communicator::world().rank() == 0 ) 
#endif
        {
            std::cout << std::string(output::linesize, '*') << '\n';
        }
    }

    void title(const std::string& str__)
    {
#ifdef EDUS_MPI
        if ( mpi::Communicator::world().rank() == 0 ) 
#endif
        {
            std::stringstream title;
            title << std::string(5, ' ') <<  str__ << std::string(5, ' ');
            int num_stars = (output::linesize - title.str().length())/2;
            std:: cout << std::string(num_stars,'*') << title.str() << std::string(num_stars,'*') << std::endl;
        }
    }

}