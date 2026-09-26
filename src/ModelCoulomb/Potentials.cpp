#include <cmath>
#include "Potentials.hpp"
#include "Constants.hpp"

namespace pot {

/// @brief Calculates the struve special function using the algorithm from XATU
/// @param X__ variable on which we calculate the struve function
/// @param v__ Order of the struve function; only 0 is allowed
/// @return H_v(X)
double struve(const double& X__, const double& v__)
{
    if (v__!=0) {
        throw std::runtime_error("struve functions are defined only for oder 0.");
    }
    double SH0;
    double A0,BY0,P0,Q0,R,S,T,T2,TA0;
	int K, KM;

    S=1.0;
    R=1.0;
    if (X__ <= 20.0) {
        A0=2.0*X__/pi;
        for (K=1; K<61; K++) {
            R=-R*X__/(2.0*K+1.0)*X__/(2.0*K+1.0);
            S=S+R;
            if (fabs(R) < fabs(S)*1.0e-12) goto e15;
        }
        e15:       SH0=A0*S;
    }
    else {
        KM=int(0.5*(X__+1.0));
        if (X__ >= 50.0) KM=25;
        for (K=1; K<=KM; K++) {
           R=-R*pow((2.0*K-1.0)/X__,2);
           S=S+R;
           if (fabs(R) < fabs(S)*1.0e-12) goto e25;
        }
        e25:       T=4.0/X__;
        T2=T*T;
        P0=((((-.37043e-5*T2+.173565e-4)*T2-.487613e-4)*T2+.17343e-3)*T2-0.1753062e-2)*T2+.3989422793;
        Q0=T*(((((.32312e-5*T2-0.142078e-4)*T2+0.342468e-4)*T2-0.869791e-4)*T2+0.4564324e-3)*T2-0.0124669441);
        TA0=X__-0.25*pi;
        BY0=2.0/sqrt(X__)*(P0*sin(TA0)+Q0*cos(TA0));
        SH0=2.0/(pi*X__)*S+BY0;
    }    
    return SH0;
}


/// @brief Calculates the usual Coulomb potential defined as @f$ V(r) = \frac{1}{\epsilon r} @f$
/// @param r__ Value on which we want to evaluate the coulomb potential
/// @param epsilon__ Macroscopic screening
/// @return V(r)
double Coulomb(const double r__, const double epsilon__)
{
    return 1./(epsilon__*r__);
}

/// @brief Calculates the Rytova Keldysh potential defined as @f$ V(r) = \frac{\pi}{2 r_0 \epsilon} \Big( H_0(r)-Y_0(r)\Big)
/// @param r__ Value on which we want to evaluate the coulomb potential
/// @param epsilon__ @f$ \epsilon @f$ in Rytova-Keldysh potential
/// @param r0__ @f$ r_0 @f$ in Rytova-Keldysh potential
/// @return @f$ V_{RK}(r) @f$
double RytovaKeldysh(const double r__, const double epsilon__, const double r0__)
{    
    return pi/(2.*r0__*epsilon__)*(struve(r__,0)-y0(r__));
}

}