#ifndef MODEL_HPP
#define MODEL_HPP


#include <cassert>
#include "ConvertUnits.hpp"
#include "core/profiler.hpp"
#include "Wannier/Wannier.hpp"

#include "Operator/Operator.hpp"


class Material
{
    private:


    public:
        enum MODEL{wannierTB} model;

        Operator<std::complex<double>> H;
        std::array< Operator<std::complex<double>>, 3> r;

        std::vector<Coordinate> rwann_;


        Material(){};
        //Material(const Material& m) = default;
        //Material& operator=(const Material& m) = default;
        
        Material(Material&& m) = default;
        Material& operator=(Material&& m) = default;

        Material(const std::string& Filename)
        {
            PROFILE("Material");
            model = wannierTB;
   
            /* read tb file */
            Wannier wannier_(Filename);
            
            /* initialize basis for the simulation */
            auto UC = Matrix<double>(wannier_.UnitCell).transpose();
            Convert_iterable(UC, Angstrom, AuLength);   
            Basis LatticeVectors_(UC);
            Coordinate::add_Basis(LatticeVectors_, LatticeVectors(R));
            Basis ReciprocalLatticeVectors(2.*pi*UC.inverse().transpose());
            Coordinate::add_Basis(ReciprocalLatticeVectors, LatticeVectors(k));
            
            /* copy wannier operators of tb file into EDUS objects */
            r[0].get_Operator_R().initialize(R, wannier_.r[0]);
            r[1].get_Operator_R().initialize(R, wannier_.r[1]);
            r[2].get_Operator_R().initialize(R, wannier_.r[2]);
            H.get_Operator_R().initialize(R, wannier_.H);

            Convert_iterable(r[0].get_Operator_R(), Angstrom, AuLength);
            Convert_iterable(r[1].get_Operator_R(), Angstrom, AuLength);
            Convert_iterable(r[2].get_Operator_R(), Angstrom, AuLength);
            Convert_iterable(H.get_Operator_R(), ElectronVolt, AuEnergy);

            r[0].get_Operator_R().Divide( wannier_.Degeneracy);
            r[1].get_Operator_R().Divide( wannier_.Degeneracy);
            r[2].get_Operator_R().Divide( wannier_.Degeneracy);
            H.get_Operator_R()   .Divide( wannier_.Degeneracy);
            /* copy wannier mesh of tb file into EDUS object */
            MeshGrid aux_mg(R, wannier_.Rmesh, LatticeVectors(R));

            H.get_Operator_R().set_MeshGrid(aux_mg);
            r[0].get_Operator_R().set_MeshGrid(aux_mg);
            r[1].get_Operator_R().set_MeshGrid(aux_mg);
            r[2].get_Operator_R().set_MeshGrid(aux_mg);

            /* initialize gauges of operators */
            r[0].lock_gauge(wannier);      r[0].lock_space(R);        
            r[1].lock_gauge(wannier);      r[1].lock_space(R);        
            r[2].lock_gauge(wannier);      r[2].lock_space(R);        
            H.lock_gauge(wannier);         H.lock_space(R);        
    
            /* assume wannier centers are the R=0 components, diagonal of r*/
            rwann_ = std::vector<Coordinate>(r[0].get_Operator(R).get_nrows());
            auto index_origin = r[0].get_Operator_R().get_MeshGrid()->find(Coordinate(0,0,0));
            auto& x0 = (r[0].get_Operator_R())[index_origin];
            auto& y0 = (r[1].get_Operator_R())[index_origin];
            auto& z0 = (r[2].get_Operator_R())[index_origin];
            for(int iwann=0; iwann<x0.get_nrows(); ++iwann) {
                rwann_[iwann] = Coordinate(x0(iwann,iwann).real(), y0(iwann,iwann).real(), z0(iwann,iwann).real());
            }
        }
};


#endif
