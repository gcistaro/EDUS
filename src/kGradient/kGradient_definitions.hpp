
/*
   This function calculates the gradient supposing you have a DerivativeFunction already allocated!
   in k space, we use the formula from Mostofi2008; in R space, we calculate it as:
   Derivative(R) = i*R*Function(R)
*/
/* number of q points of the function: BlockMatrix has get_nq(), other containers (mdarray) have nq = 1 */
template<typename T>
auto kgradient_get_nq(const T& function__, int) -> decltype(function__.get_nq())
{
    return function__.get_nq();
}

template<typename T>
int kgradient_get_nq(const T&, long)
{
    return 1;
}

template<typename T, typename U, typename S>
void kGradient::Calculate(const S& scalar, T& DerivativeFunction, const T& Function, 
                          const U& direction, const bool& EraseOutput) const
{
#ifdef EDUS_TIMERS
    PROFILE("kGradient::Calculate");
#endif
    if( EraseOutput ) {
        DerivativeFunction.fill(0.);
    }

    /* case 1-> we calculate the gradient directly in k */
    const int nq = kgradient_get_nq(Function, 0);

    if( kmesh && !Rmesh ) {
        if( nq > 1 ) {
            throw std::runtime_error("kGradient::Calculate in k: not implemented for nq > 1, use gradient_space = R\n");
        }
        #pragma omp parallel for schedule(static)
        for( int ik = 0; ik < mpindex.nlocal; ++ik ) {
            for( int ishell = 0; ishell < Weight.get_Size(0); ++ishell ) {
                for( int ib = 0; ib < int( ikshell[ishell].size() ); ++ib ) {
                    auto& Bvector = (*kmesh)[ikshell[ishell][ib]];
                    auto bdotu = Bvector.dot(direction);
                    auto& ikpb_ = ikpb[ik][ishell][ib];
                    DerivativeFunction[ik] += scalar*Weight(ishell)*bdotu*Function[ikpb_];  
                }
            }
        }
    }
    /* case 2-> we calculate the gradient in R. It is up to the user to make sure Function
       is defined in R and to go back to k after doing it. TODO!! Change this!!
    */
    else if ( Rmesh && !kmesh ) {
        #pragma omp parallel for schedule(static)
        for( int iR_loc = 0; iR_loc < mpindex.nlocal; ++iR_loc ) {
            auto iR_global = (const_cast<MPIindex<3>&>(mpindex)).loc1D_to_glob1D(iR_loc);
            auto factor = scalar*im*(*Rmesh)[iR_global].dot(direction);
            /* at fixed q, R is conjugate to k: the same factor i*R.direction for every q */
            for( int iq = 0; iq < nq; ++iq ) {
                const int iblock = iR_loc*nq + iq;
                DerivativeFunction[iblock] += factor*Function[iblock];
            }
        }
    }
}