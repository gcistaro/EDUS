/// @brief Standard function defining the initial density matrix.
/// The initial density matrix is the equilibrium one, computed in electron::System:
/// @f[
/// \rho_{nn}(\textbf{k}) = f_n \quad \text{in the bloch gauge}
/// @f]
/// already rotated to the wannier gauge, where the equations are propagated.
/// @param DM__ The Operator where we want to store the initial density matrix
std::function<void(Operator<std::complex<double>>&)>
InitialCondition =
[&](Operator<std::complex<double>>& DM__)
{
    PROFILE("RK::InitialCondition");
    auto& DM0k = electrons_.DM0().get_Operator(Space::k);
    std::copy(DM0k.begin(), DM0k.end(), DM__.get_Operator_k().begin());

    DM__.lock_gauge(wannier);
    DM__.lock_space(k);

    if(SpaceOfPropagation_ == R) DM__.go_to_R();
};
