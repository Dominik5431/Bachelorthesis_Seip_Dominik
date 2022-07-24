using LinearAlgebra
using Random
using QuantumInformation
using Optim
using Plots



b2 = Array{Complex{Float64}}(undef, 2, 2, 4)

b2[:,:,1] = [1 0; 0 1]/sqrt(2)
b2[:,:,2] = [0 1; 1 0]/sqrt(2)
b2[:,:,3] = [0 -1im; 1im 0]/sqrt(2)
b2[:,:,4] = [1 0; 0 -1]/sqrt(2)

b3 = Array{Complex{Float64}}(undef, 3, 3, 9)

b3[:,:,1] = [1 0 0; 0 1 0; 0 0 1]/sqrt(3)
b3[[1 2],[1 2], 2] = b2[:,:,2]
b3[[1 2],[1 2], 3] = b2[:,:,3]
b3[[1 2],[1 2], 4] = b2[:,:,4]
b3[[1 3],[1 3], 5] = b2[:,:,2]
b3[[1 3],[1 3], 6] = b2[:,:,3]
b3[[2 3],[2 3], 7] = b2[:,:,2]
b3[[2 3],[2 3], 8] = b2[:,:,3]
b3[:,:, 9] = [1 0 0; 0 1 0; 0 0 -2]/sqrt(6)

x_b6 = Array{Complex{Float64}}(undef, 6, 6, 11)

x_b6[:,:,1] = kron(b2[:,:,1], b3[:,:,4])
x_b6[:,:,2] = kron(b2[:,:,1], b3[:,:,9])
x_b6[:,:,3] = kron(b2[:,:,4], b3[:,:,1])
x_b6[:,:,4] = kron(b2[:,:,2], b3[:,:,5])
x_b6[:,:,5] = kron(b2[:,:,2], b3[:,:,6])
x_b6[:,:,6] = kron(b2[:,:,3], b3[:,:,5])
x_b6[:,:,7] = kron(b2[:,:,3], b3[:,:,6])
x_b6[:,:,8] = kron(b2[:,:,4], b3[:,:,4])
x_b6[:,:,9] = kron(b2[:,:,4], b3[:,:,9])
x_b6[:,:,10] = 1/sqrt(6) * kron(b2[:,:,2], b3[:,:,9]) + 1/sqrt(3) * kron(b2[:,:,2], b3[:,:,1]) - 1/sqrt(2) * kron(b2[:,:,2], b3[:,:,4])
x_b6[:,:,11] = 1/sqrt(6) * kron(b2[:,:,3], b3[:,:,9]) + 1/sqrt(3) * kron(b2[:,:,3], b3[:,:,1]) - 1/sqrt(2) * kron(b2[:,:,3], b3[:,:,4])

spin1 = Array{Complex{Float64}}(undef, 3, 3, 3)
spin1[:,:,1] = 1/sqrt(2) * [0 1 0;1 0 1;0 1 0]
spin1[:,:,2] = 1/sqrt(2) * [0 -1im 0;1im 0 -1im;0 1im 0]
spin1[:,:,3] = [1 0 0;0 0 0;0 0 -1]


function getRandomUnitVector(n)
    vec = real(randn(n))
    while norm(vec) < 0.0001
        vec = real(randn(n))
    end
    return vec/norm(vec)
end

function rotation2(phi)
    [cos(phi) -sin(phi) ; sin(phi) cos(phi)]
end

function rotation3(phi, axis)
    if axis == 1
        [cos(phi) -sin(phi) 0 ; sin(phi) cos(phi) 0 ; 0 0 1]
    elseif axis == 2
        [cos(phi) 0 -sin(phi) ; 0 1 0 ; sin(phi) 0 cos(phi)]
    elseif axis == 3
        [1 0 0 ; 0 cos(phi) -sin(phi) ; 0 sin(phi) cos(phi)]
    end
end

function getNextState(state)
    dir = getRandomUnitVector(11)  
    vmax =  2 * sqrt(5 / 6)
    vmin = -2 * sqrt(5 / 6)
    v = vmin + (vmax-vmin)*rand()
    a = state + v * dir
    while !is_densitymatrix(make_densitymatrix(a))
        (v < 0) ? (vmin = v) : (vmax = v)
        v = vmin + (vmax - vmin) * rand()
        a = state + v * dir
    end 
    rad = norm(a)/sqrt(5/6)
    return a, rad
end

function Base.run(n::Int)
    n_ppt = 0
    n_maj = 0
    n_bell = 0 #amount of states which violate the Bell inequality
    n_CG = 0
    radius = zeros(Int64, 100)
    radius_ppt = zeros(Int64, 100)
    radius_maj = zeros(Int64, 100)
    radius_bell = zeros(Int64, 100)
    radius_CG = zeros(Int64,100)
    vec = zeros(Float64, 11)
    @time Threads.@threads for i in 1:n
        state = getNextState(vec)
        vec = state[1]
        rad = state[2]
        #println(rad)
        state = make_densitymatrix(vec)
        #display(eigvals(state))
        if is_densitymatrix(state)  
            radius[Int8(round(rad*50, RoundDown))+1] += 1
            if PPT_criterion(state)
                n_ppt += 1
                radius_ppt[Int8(round(rad*50, RoundDown))+1] += 1
                #=if violates_CG(state)
                    n_CG += 1
                    radius_CG[Int8(round(rad*50, RoundDown))+1] += 1
                end 
                if violates_bell(state)
                    n_bell += 1
                    radius_bell[Int8(round(rad*50, RoundDown))+1] += 1
                end=#
            #else
            if violates_CG(state)
                n_CG += 1
                radius_CG[Int8(round(rad*50, RoundDown))+1] += 1
            end
            end
            if violates_bell(state)
                n_bell += 1
                radius_bell[Int8(round(rad*50, RoundDown))+1] += 1
            end
            if majorization_criterion(state)
                n_maj += 1
                radius_maj[Int8(round(rad*50, RoundDown))+1] += 1
            end
        end
    end
    return n_ppt/n, n_maj/n, n_bell/n, n_CG/n, radius, radius_ppt, radius_maj, radius_bell, radius_CG
end


function dagger(x)
    return transpose(conj(x))
end

function getTrho(rho)
    C = zeros(Complex{Float64},(3,3))
    for i in 1:3
        for j in 1:3
            C[i,j] = tr(rho * kron(sqrt(2) * b2[:,:,i+1], proje[:,:,j,1]-proje[:,:,j,2]-proje[:,:,j,3]))
        end
    end
    return C
end

function getVecA(rho)
    C = zeros(Complex{Float64},3)
    for i in 1:3    
        C[i] = sqrt(2) * sqrt(3) * tr(rho * kron(b2[:,:,i+1], b3[:,:,1]))
    end
    return C
end

function getVecB(rho)
    C = zeros(Complex{Float64},3)
    for i in 1:3    
        C[i] = sqrt(2) * tr(rho * kron(b2[:,:,1], proje[:,:,i,1]-proje[:,:,i,2]-proje[:,:,i,3]))
    end
    return C
end


function sphere_vector(phi, theta)
    return [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]
end

function dagger(x)
    return transpose(conj(x))
end

proje = Array{Complex{Float64}}(undef, 3, 3, 3, 3)

proje[:,:,1,3] = eigvecs(spin1[:,:,1])[:,1] * transpose(eigvecs(spin1[:,:,1])[:,1])
proje[:,:,1,2] = eigvecs(spin1[:,:,1])[:,2] * transpose(eigvecs(spin1[:,:,1])[:,2])
proje[:,:,1,1] = eigvecs(spin1[:,:,1])[:,3] * transpose(eigvecs(spin1[:,:,1])[:,3])

proje[:,:,2,3] = eigvecs(spin1[:,:,2])[:,1] * conj(transpose(eigvecs(spin1[:,:,2])[:,1]))
proje[:,:,2,2] = eigvecs(spin1[:,:,2])[:,2] * conj(transpose(eigvecs(spin1[:,:,2])[:,2]))
proje[:,:,2,1] = eigvecs(spin1[:,:,2])[:,3] * conj(transpose(eigvecs(spin1[:,:,2])[:,3]))

proje[:,:,3,3] = eigvecs(spin1[:,:,3])[:,1] * transpose(eigvecs(spin1[:,:,3])[:,1])
proje[:,:,3,2] = eigvecs(spin1[:,:,3])[:,2] * transpose(eigvecs(spin1[:,:,3])[:,2])
proje[:,:,3,1] = eigvecs(spin1[:,:,3])[:,3] * transpose(eigvecs(spin1[:,:,3])[:,3])

function clear(mat)
    return broadcast(x -> (abs(x) < 1e-15) ? 0 : ((abs(real(x))<1e-15 ? 0 : real(x)) + 1im * (abs(imag(x))<1e-15 ? 0 : imag(x))) , mat)
end


function violates_bell(rho)
    C = transpose(getTrho(rho))
    ev = eigvals(dagger(C) * C)
    ev_sort = sort(ev, rev = true)
    return ev_sort[1] + ev_sort[2] > 1 + 1e-14
end

function violates_CG(rho)
    tauA = real(getVecA(rho))
    tauB = real(getVecB(rho))
    Crho = transpose(real(getTrho(rho)))
    ineq(x) = 4 + dot(sphere_vector(x[1],x[2])+sphere_vector(x[3],x[4]),tauB) - norm(Crho * (sphere_vector(x[1],x[2])+sphere_vector(x[3],x[4])+sphere_vector(x[5],x[6])) + tauA) - norm(Crho * (sphere_vector(x[1],x[2])+sphere_vector(x[3],x[4])-sphere_vector(x[5],x[6])) + tauA) - norm(Crho * (sphere_vector(x[1],x[2])-sphere_vector(x[3],x[4])))
    result = give_minimum(ineq)
    return result <= 1e-15
end

function give_minimum(f)
    res = optimize(f,[0.0,0.0,0.0,0.0,0.0,0.0])
    Optim.minimum(res)
end

function make_densitymatrix(x)
    result = 1/6 * Matrix(I,6,6)
    for i in 1:11
        result += x[i] * x_b6[:,:,i] 
    end
    return result
end

function matrixpower(x, n)
    if n==1 || n==0
        return x
    else
        return x * matrixpower(x, n-1)
    end
end

function is_densitymatrix(x)
    #if real(tr(x))-1 > 1e-10
    #    throw(ArgumentError("Trace not equal one!"))
    #end
    if abs(maximum(real(transpose(conj(x)) - x))) > 1e-14 || abs(maximum(real(x - transpose(conj(x))))) > 1e-14 
        throw(ArgumentError("Matrix is not hermitian!"))
    end 
    p2 = real(tr(matrixpower(x,2)))
    p3 = real(tr(matrixpower(x,3)))
    p4 = real(tr(matrixpower(x,4)))
    p5 = real(tr(matrixpower(x,5)))
    p6 = real(tr(matrixpower(x,6)))
    if newton_2(p2) && newton_3(p2, p3) && newton_4(p2, p3, p4) && newton_5(p2, p3, p4, p5) && newton_6(p2, p3, p4, p5, p6)
        return true
    else
        return false
    end
end

#Newton-identity 1 is trivial fullfilled
function newton_2(p2)
    return (1 - p2 >= 0 + 1e-14)
end

function newton_3(p2,p3)
    return (1 - 3 * p2 + 2 * p3 >= 0 + 1e-14)
end

function newton_4(p2,p3,p4)
    return (1 - 6 * p2 + 8 * p3 + 3 * p2^2 - 6 * p4 >= 0 + 1e-14)
end

function newton_5(p2,p3,p4,p5)
    return (1 - 10 * p2 + 20 * p3- 30 * p4 + 24*p5 + 15 * p2^2 - 20*p2*p3  >= 0 + 1e-14 )
end

function newton_6(p2,p3,p4,p5,p6)
    return (1- 15 * p2 + 40 * p3 - 90 * p4 + 144*p5 - 120*p6
             + 45 * p2^2 - 120*p2*p3 + 90*p2*p4 + 40*p3^2 - 15*p2^3  >= 0 + 1e-14 )
end

#https://github.com/jump-dev/Convex.jl/blob/cba179feeaa7934e0e70011ef74c181a7e88ccbd/src/atoms/affine/partialtranspose.jl#L44-L69
#method ptranspose() approx half speed of execution
function partialtranspose(x::AbstractMatrix, sys::Int, dims::Vector) 
    if size(x,1) ≠ size(x,2)
            throw(ArgumentError("Only square matrices are supported"))
    end
    if ! (1 ≤ sys ≤ length(dims))
            throw(ArgumentError("Invalid system, should between 1 and $(length(dims)); got $sys"))
    end
    if size(x,1) ≠ prod(dims)
            throw(ArgumentError("Dimension of system doesn't correspond to dimension of subsystems"))
    end
    n = length(dims)
    d = prod(dims)
    s = n - sys + 1
    p = collect(1:2n)
    p[s] = n + s
    p[n + s] = s

    rdims = reverse(dims)
    r = reshape(x, (rdims..., rdims...))
    temp = permutedims(r,p)
    result = reshape(permutedims(r,p),(d,d))
    return reshape(permutedims(r,p),(d,d))
end

function PPT_criterion(x)
    return is_densitymatrix(partialtranspose(x, 2, [2,3]))
end

function majorization_criterion(x)
    return (maj_eigenvalues(x, ptrace(x,[2,3],1)) && maj_eigenvalues(x, ptrace(x,[2,3],2)))
end

function maj_eigenvalues(x,y)
    eig_x = sort(broadcast(abs, eigvals(x)), rev=true)
    eig_y = sort(broadcast(abs, eigvals(y)), rev=true)
    lx = length(eig_x)
    ly = length(eig_y)
    if lx<ly
        eig_x = vcat(eig_x, zeros(ly-lx))
        lx=length(eig_x)
    elseif ly<lx
        eig_y = vcat(eig_y, zeros(lx-ly))
        ly=length(eig_y)
    end
    for i in 1:length(eig_x)-1
        if sum(eig_x[1:i]) - sum(eig_y[1:i]) > -1e-15
            return false
        end
    end
    if abs(sum(eig_x) - sum(eig_y)) > 1e-14
        return false
    end
    return true
end

function reduction_criterion(x)
    px = kron(ptrace(x, [2,3], 2), Matrix(I,3,3)) - x
    ev = real(eigvals(px))    
    return all(>=(0), ev) 
end



phi1 = 0
phi2 = 0
axis = 1


function final_run(n::Int, rep::Int)
    resPPT = zeros(rep)
    resMaj = zeros(rep)
    resBell = zeros(rep)
    resCG = zeros(rep)
    radTot = zeros(rep,100)
    radPPT = zeros(rep,100)
    radMaj = zeros(rep,100) 
    radBell = zeros(rep,100)
    radCG = zeros(rep,100) 
    for i in 1:rep
        resPPT[i], resMaj[i], resBell[i], resCG[i], radTot[i,:], radPPT[i,:], radMaj[i,:], radBell[i,:], radCG[i,:] = run(n)
    end
    return resPPT, resMaj, resBell, resCG, radTot, radPPT, radMaj, radBell, radCG
end

final_run(1000,1)
@time res = final_run(5000000, 2)


function mu_sigma(arr)
    mean = sum(arr)/length(arr)
    sigma = 0
    for i in 1:length(arr)
        sigma += (arr[i]-mean)^2
    end
    sigma = 1/(length(arr)-1) * sigma
    sigma = sqrt(sigma)
    return mean, sigma
end



