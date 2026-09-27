/// EDUS_wfc2wannier: reads the Kohn-Sham wavefunctions of Quantum ESPRESSO and rotates them
/// to the Wannier gauge using the matrices written by wannier90 (write_u_matrices = .true.).
///
/// Usage: EDUS_wfc2wannier input.json
///
/// {
///   "qe_save_dir"  : "out/si.save",      // required, contains wfc<ik>.dat
///   "seedname"     : "wannier/si",       // required, <seedname>_u.mat [, _u_dis.mat, .eig]
///   "exclude_bands": [1,2,3,4,5],        // as in the .win file (1-based)
///   "dis_win_min"  : -100.0,             // outer window (eV), only if used in the .win file
///   "dis_win_max"  : 20.0,
///   "spin"         : "none",             // none | up | down (LSDA)
///   "fft_grid"     : [0,0,0],            // 0 = automatic
///   "supercell"    : [3,3,3],
///   "wannier_list" : [],                 // 1-based, empty = all
///   "write_bloch"  : true,               // <output_dir>/wfc<ik>.dat in QE format, num_wann bands
///   "real_space"   : true,               // build w_n(r) on the supercell
///   "write_xsf"    : true,
///   "fix_phase"    : true,
///   "output_dir"   : "wannier_wfc"
/// }
#include <fstream>
#include <iostream>

#include "Json/json.hpp"
#include "Wannier/WannierWavefunctions.hpp"
#include "core/print_timing.hpp"
#include "core/profiler.hpp"
#include "initialize.hpp"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: EDUS_wfc2wannier <input.json>\n";
        return 1;
    }
    initialize();
    {
        PROFILE("wfc2wannier");
        std::ifstream f(argv[1]);
        if (!f.is_open()) {
            throw std::runtime_error(std::string("cannot open ") + argv[1]);
        }
        auto json   = nlohmann::json::parse(f);
        auto params = WannierWavefunctionsParameters::from_json(json);
        WannierWavefunctions wwf(params);
        wwf.run();
    }
    finalize();
    return 0;
}
