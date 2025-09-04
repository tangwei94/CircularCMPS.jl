
# transfer matrix in the space (ℂ^χ)^d ⊗ (ℂ^χ)'^d. deprecated 
#function transfer_matrix(ϕ::MultiBosonCMPSData, ψ::MultiBosonCMPSData)
#    d = get_d(ψ)
#    function fK(v)
#        Tv = ψ.Q * v + v * ϕ.Q'
#        for ix in 1:d
#            Md, Mu = ψ.Ms[ix], ϕ.Ms[ix]
#            vindices = Vector(-1:-1:(-2*d)) 
#            vindices[ix] = 1
#            vindices[ix+d] = 2
#            Tv += permute(ncon([Md, v, Mu'], [[-ix, 1], vindices, [2, -ix-d]]), Tuple(1:d), Tuple(d+1:2d));
#        end
#        return Tv
#    end
#    return fK 
#end
#
#"""
#   transfer_matrix_dagger(ϕ::CMPSData{S}, ψ::CMPSData{S}) where S
#    
#   The Hermitian conjugate of the transfer matrix for <ϕ|ψ>.
#   target at left vector. 
#   T† = I + ϵK†. returns K†
#"""
#function transfer_matrix_dagger(ϕ::MultiBosonCMPSData, ψ::MultiBosonCMPSData) 
#    d = get_d(ψ)
#    function fK_dagger(v)
#        Tv = v * ψ.Q + ϕ.Q' * v 
#        for ix in 1:d
#            Md, Mu = ψ.Ms[ix], ϕ.Ms[ix]
#            vindices = Vector(-1:-1:(-2*d)) 
#            vindices[ix] = 1
#            vindices[ix+d] = 2
#            Tv += permute(ncon([Mu', v, Md], [[-ix, 1], vindices, [2, -ix-d]]), Tuple(1:d), Tuple(d+1:2d));
#        end
#        return Tv
#    end
#    return fK_dagger
#end
