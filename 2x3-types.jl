using LinearAlgebra
using Random
using QuantumInformation
using Optim

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

k36 = Array{Complex{Float64}}(undef, 6, 6, 36)
for i in 0:3
    for j in 1:9
        k36[:,:,9*i+j] = kron(b2[:,:,i+1],b3[:,:,j])
    end    
end
k35 = k36[:,:,2:36]

k35p = Array{Complex{Float64}}(undef, 6, 6, 35)
function initialize_k35p()
    k35p[:,:,1] = 1/sqrt(2) * (k35[:,:,1] + k35[:,:,28])
    k35p[:,:,2] = 1/sqrt(2) * (k35[:,:,1] - k35[:,:,28])
    k35p[:,:,3] = 1/sqrt(2) * (k35[:,:,2] + k35[:,:,29])
    k35p[:,:,4] = 1/sqrt(2) * (k35[:,:,2] - k35[:,:,29])
    k35p[:,:,5] = 1/sqrt(2) * (k35[:,:,13] + k35[:,:,23])
    k35p[:,:,6] = 1/sqrt(2) * (k35[:,:,13] - k35[:,:,23])
    k35p[:,:,7] = 1/sqrt(2) * (k35[:,:,14] + k35[:,:,22])
    k35p[:,:,8] = 1/sqrt(2) * (k35[:,:,14] - k35[:,:,22])
    k35p[:,:,9] = 1/sqrt(2) * (k35[:,:,6] + k35[:,:,33])
    k35p[:,:,10] = 1/sqrt(2) * (k35[:,:,6] - k35[:,:,33])
    k35p[:,:,11] = 1/sqrt(2) * (k35[:,:,7] + k35[:,:,34])
    k35p[:,:,12] = 1/sqrt(2) * (k35[:,:,7] - k35[:,:,34])
    k35p[:,:,13] = 1/sqrt(2) * (k35[:,:,4] + k35[:,:,31])
    k35p[:,:,14] = 1/sqrt(2) * (k35[:,:,4] - k35[:,:,31])
    k35p[:,:,15] = 1/sqrt(2) * (k35[:,:,5] + k35[:,:,32])
    k35p[:,:,16] = 1/sqrt(2) * (k35[:,:,5] - k35[:,:,32])
    k35p[:,:,17] = 1/sqrt(2) * (k35[:,:,15] + k35[:,:,25])
    k35p[:,:,18] = 1/sqrt(2) * (k35[:,:,15] - k35[:,:,25])
    k35p[:,:,19] = 1/sqrt(2) * (k35[:,:,16] + k35[:,:,24])
    k35p[:,:,20] = 1/sqrt(2) * (k35[:,:,16] - k35[:,:,24])
    k35p[:,:,21] = 1/sqrt(2) * (k35[:,:,10] + k35[:,:,20])
    k35p[:,:,22] = 1/sqrt(2) * (k35[:,:,10] - k35[:,:,20])
    k35p[:,:,23] = 1/sqrt(2) * (k35[:,:,11] + k35[:,:,19])
    k35p[:,:,24] = 1/sqrt(2) * (k35[:,:,11] - k35[:,:,19])
    k35p[:,:,25:29] = k35[:,:,[3,8,27,30,35]]
    k35p[:,:,30] = 1/sqrt(3) * (k35[:,:,9] - sqrt(2) * k35[:,:,17])
    k35p[:,:,31] = 1/sqrt(3) * (k35[:,:,18] - sqrt(2) * k35[:,:,26])
    k35p[:,:,32] = 1/sqrt(6) * k35[:,:,17] + 1/sqrt(3) * k35[:,:,9] - 1/sqrt(2) * k35[:,:,12]
    k35p[:,:,33] = 1/sqrt(6) * k35[:,:,17] + 1/sqrt(3) * k35[:,:,9] + 1/sqrt(2) * k35[:,:,12]
    k35p[:,:,34] = 1/sqrt(6) * k35[:,:,26] + 1/sqrt(3) * k35[:,:,18] - 1/sqrt(2) * k35[:,:,21]
    k35p[:,:,35] = 1/sqrt(6) * k35[:,:,26] + 1/sqrt(3) * k35[:,:,18] + 1/sqrt(2) * k35[:,:,21]
end

initialize_k35p()

function commutator(x,y)
    return x*y - y*x
end

function clear(mat)
    return broadcast(x -> (abs(x) < 1e-15) ? 0 : ((abs(real(x))<1e-15 ? 0 : real(x)) + 1im * (abs(imag(x))<1e-15 ? 0 : imag(x))) , mat)
end

x_b6 = Array{Complex{Float64}}(undef, 6, 6, 11,15)

function initialize_basis()
    x_b6[:,:,:,1] = k35p[:,:,[25,26,27,28,29,5,6,7,8,32,34]] 

    x_b6[:,:,1:5,2] = k35[:,:,[3,8,27,30,35]]
    x_b6[:,:,6,2] = 1/sqrt(2) * (k35[:,:,1] + k35[:,:,28])
    x_b6[:,:,7,2] = 1/sqrt(2) * (k35[:,:,2] + k35[:,:,29])
    x_b6[:,:,8,2] = 1/sqrt(2) * (k35[:,:,13] + k35[:,:,23])
    x_b6[:,:,9,2] = 1/sqrt(2) * (k35[:,:,22] - k35[:,:,14])
    x_b6[:,:,10,2] = 1/sqrt(2) * (k35[:,:,6] - k35[:,:,33])
    x_b6[:,:,11,2] = 1/sqrt(2) * (k35[:,:,7] - k35[:,:,34])

    #x_b6[:,:,1:9,6] = k35[:,:,[3,8,27,30,35,4,5,31,32]]
    x_b6[:,:,1:9,6] = k35p[:,:,[25,26,27,28,29,13,14,15,16]]
    x_b6[:,:,10,6] = 1/sqrt(6) * kron(b2[:,:,2], b3[:,:,9]) + 1/sqrt(3) * kron(b2[:,:,2], b3[:,:,1]) - 1/sqrt(2) * kron(b2[:,:,2], b3[:,:,4])
    x_b6[:,:,11,6] = 1/sqrt(6) * kron(b2[:,:,3], b3[:,:,9]) + 1/sqrt(3) * kron(b2[:,:,3], b3[:,:,1]) - 1/sqrt(2) * kron(b2[:,:,3], b3[:,:,4])

    x_b6[:,:,1:5,7] = k35[:,:,[3,8,27,30,35]]
    x_b6[:,:,6,7] = 1/sqrt(2) * (k35[:,:,4] + k35[:,:,31])
    x_b6[:,:,7,7] = 1/sqrt(2) * (k35[:,:,5] + k35[:,:,32])
    x_b6[:,:,8,7] = k35p[:,:,18]
    x_b6[:,:,9,7] = k35p[:,:,19]
    x_b6[:,:,10,7] = k35p[:,:,2]
    x_b6[:,:,11,7] = k35p[:,:,4]

    #x_b6[:,:,1:9,5] = k35[:,:,[3,8,27,30,35,4,5,31,32]]
    x_b6[:,:,:,5] = k35p[:,:,[25,26,27,28,29,13,15,21,24,10,12]]
   

    x_b6[:,:,1:5,3] = k35[:,:,[3,8,27,30,35]]
    x_b6[:,:,6,3] = 1/sqrt(2) * (k35[:,:,1] + k35[:,:,28])
    x_b6[:,:,7,3] = 1/sqrt(2) * (k35[:,:,2] + k35[:,:,29])
    x_b6[:,:,8,3] = 1/sqrt(2) * (k35[:,:,15] + k35[:,:,25])
    x_b6[:,:,9,3] = 1/sqrt(2) * (k35[:,:,16] - k35[:,:,24])
    x_b6[:,:,10,3] = 1/sqrt(2) * (k35[:,:,4] - k35[:,:,31])
    x_b6[:,:,11,3] = 1/sqrt(2) * (k35[:,:,5] - k35[:,:,32])

    x_b6[:,:,1:5,4] = k35[:,:,[3,8,27,30,35]]
    x_b6[:,:,6,4] = 1/sqrt(2) * (k35[:,:,1] + k35[:,:,28])
    x_b6[:,:,7,4] = 1/sqrt(2) * (k35[:,:,2] + k35[:,:,29])
    x_b6[:,:,8,4] = 1/sqrt(2) * (k35[:,:,1] - k35[:,:,28])
    x_b6[:,:,9,4] = 1/sqrt(2) * (k35[:,:,2] - k35[:,:,29])
    x_b6[:,:,10,4] = 1/sqrt(3) * (k35[:,:,9] - sqrt(2) * k35[:,:,17])
    x_b6[:,:,11,4] = 1/sqrt(3) * (k35[:,:,18] - sqrt(2) * k35[:,:,26])


    x_b6[:,:,1:9,8] = k35p[:,:,[25,26,27,28,29,9,10,11,12]]
    x_b6[:,:,10,8] = 1/sqrt(3) * (k35[:,:,9] + 1/sqrt(2) * k35[:,:,17] + sqrt(6)/2 * k35[:,:,12])
    x_b6[:,:,11,8] = 1/sqrt(3) * (k35[:,:,18] + 1/sqrt(2) * k35[:,:,26] + sqrt(6)/2 * k35[:,:,21])

    x_b6[:,:,:,9] = k35p[:,:,[25,26,27,28,29,30,31,32,33,34,35]]

    #x_b6[:,:,1:9,10] = k35[:,:,[3,8,27,30,35,16,15,24,25]]
    x_b6[:,:,1:9,10] = k35p[:,:,[25,26,27,28,29,17,18,19,20]]
    x_b6[:,:,10,10] = 1/sqrt(3) * (k35[:,:,9] + 1/sqrt(2) * k35[:,:,17] + sqrt(6)/2 * k35[:,:,12])
    x_b6[:,:,11,10] = 1/sqrt(3) * (k35[:,:,18] + 1/sqrt(2) * k35[:,:,26] + sqrt(6)/2 * k35[:,:,21])

    x_b6[:,:,1:5,11] = k35[:,:,[3,8,27,30,35]]
    x_b6[:,:,6,11] = 1/sqrt(2) * (k35[:,:,10] - k35[:,:,20])
    x_b6[:,:,7,11] = 1/sqrt(2) * (k35[:,:,11] + k35[:,:,19])
    x_b6[:,:,8,11] = 1/sqrt(2) * (k35[:,:,6] + k35[:,:,33])
    x_b6[:,:,9,11] = 1/sqrt(2) * (k35[:,:,7] + k35[:,:,34])
    x_b6[:,:,10,11] = 1/sqrt(2) * (k35[:,:,4] - k35[:,:,31])
    x_b6[:,:,11,11] = 1/sqrt(2) * (k35[:,:,5] - k35[:,:,32])

    x_b6[:,:,1:5,12] = k35[:,:,[3,8,27,30,35]]
    x_b6[:,:,6,12] = 1/sqrt(2) * (k35[:,:,10] - k35[:,:,20])
    x_b6[:,:,7,12] = 1/sqrt(2) * (k35[:,:,11] + k35[:,:,19])
    x_b6[:,:,8,12] = 1/sqrt(2) * (k35[:,:,10] + k35[:,:,20])
    x_b6[:,:,9,12] = 1/sqrt(2) * (k35[:,:,11] - k35[:,:,19])
    x_b6[:,:,10,12] = 1/sqrt(3) * (k35[:,:,9] - sqrt(2) * k35[:,:,17])
    x_b6[:,:,11,12] = 1/sqrt(3) * (k35[:,:,18] - sqrt(2) * k35[:,:,26])

    x_b6[:,:,1:5,13] = k35[:,:,[3,8,27,30,35]]
    x_b6[:,:,6,13] = 1/sqrt(2) * (k35[:,:,10] - k35[:,:,20])
    x_b6[:,:,7,13] = 1/sqrt(2) * (k35[:,:,11] + k35[:,:,19])
    x_b6[:,:,8,13] = 1/sqrt(2) * (k35[:,:,15] - k35[:,:,25])
    x_b6[:,:,9,13] = 1/sqrt(2) * (k35[:,:,16] + k35[:,:,24])
    x_b6[:,:,10,13] = 1/sqrt(2) * (k35[:,:,13] + k35[:,:,23])
    x_b6[:,:,11,13] = 1/sqrt(2) * (k35[:,:,22] - k35[:,:,14])

    x_b6[:,:,1:5,14] = k35[:,:,[3,8,27,30,35]]
    x_b6[:,:,6,14] = 1/sqrt(2) * (k35[:,:,13] - k35[:,:,23])
    x_b6[:,:,7,14] = 1/sqrt(2) * (k35[:,:,22] + k35[:,:,14])
    x_b6[:,:,8,14] = 1/sqrt(2) * (k35[:,:,6] + k35[:,:,33])
    x_b6[:,:,9,14] = 1/sqrt(2) * (k35[:,:,7] + k35[:,:,34])
    x_b6[:,:,10,14] = 1/sqrt(2) * (k35[:,:,1] - k35[:,:,28])
    x_b6[:,:,11,14] = 1/sqrt(2) * (k35[:,:,2] - k35[:,:,29])

    x_b6[:,:,1:5,15] = k35[:,:,[3,8,27,30,35]]
    x_b6[:,:,6,15] = 1/sqrt(2) * (k35[:,:,13] - k35[:,:,23])
    x_b6[:,:,7,15] = 1/sqrt(2) * (k35[:,:,22] + k35[:,:,14])
    x_b6[:,:,8,15] = 1/sqrt(2) * (k35[:,:,11] - k35[:,:,19])
    x_b6[:,:,9,15] = 1/sqrt(2) * (k35[:,:,10] + k35[:,:,20])
    x_b6[:,:,10,15] = 1/sqrt(2) * (k35[:,:,15] + k35[:,:,25])
    x_b6[:,:,11,15] = 1/sqrt(2) * (k35[:,:,16] - k35[:,:,24])
end

initialize_basis()

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




function dagger(x)
    return transpose(conj(x))
end


#=spin1 = Array{Complex{Float64}}(undef, 3, 3, 3)
spin1[:,:,1] = sqrt(2) * [0 1 0;1 0 1;0 1 0] 
spin1[:,:,2] = sqrt(2) * [0 -1im 0;1im 0 -1im;0 1im 0]
spin1[:,:,3] = 2 * [1 0 0;0 0 0;0 0 -1]=#

spin1 = Array{Complex{Float64}}(undef, 3, 3, 3)
spin1[:,:,1] = 1/sqrt(2) * [0 1 0;1 0 1;0 1 0] 
spin1[:,:,2] = 1/sqrt(2) * [0 -1im 0;1im 0 -1im;0 1im 0]
spin1[:,:,3] = [1 0 0;0 0 0;0 0 -1]



function makeUnitary(a,n)
    if n == 2
        return exp(1im*a[1]) * exp((a[2] * sigx + a[3] * sigy + a[4] * sigz) * 1im)
    elseif n == 3
        return exp(1im*a[1]) * exp((a[2] * gamma1 + a[3] * gamma2 + a[4] * gamma3 + a[5] * gamma4 + a[6] * gamma5 + a[7] * gamma6 + a[8] * gamma7 + a[9] * gamma8) * 1im)
    end
end

function isUnitary(mat,n)
    return norm(dagger(mat) * mat - Matrix(I,n,n)) < 1e-14
end


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
        [cos(phi) sin(phi) 0 ; -sin(phi) cos(phi) 0 ; 0 0 1]
    elseif axis == 2
        [cos(phi) 0 sin(phi) ; 0 1 0 ; -sin(phi) 0 cos(phi)]
    elseif axis == 3
        [1 0 0 ; 0 cos(phi) sin(phi) ; 0 -sin(phi) cos(phi)]
    end
end

function getNextState(state, base)
    dir = getRandomUnitVector(11)  
    vmax =  2 * sqrt(5 / 6)
    vmin = -2 * sqrt(5 / 6)
    v = vmin + (vmax-vmin)*rand()
    a = state + v * dir
    while !is_densitymatrix(make_densitymatrix(a, base))
        (v < 0) ? (vmin = v) : (vmax = v)
        v = vmin + (vmax - vmin) * rand()
        a = state + v * dir
    end 
    rad = norm(a)/sqrt(5/6)
    return a, rad
end

function Base.run(n::Int, base)
    n_ppt = 0
    n_maj = 0
    n_red = 0
    n_tot = 0
    vec = zeros(Float64, 11)
    radius = zeros(Int64, 100)
    radius_ppt = zeros(Int64, 100)
    radius_maj = zeros(Int64, 100)
    radius_bell = zeros(Int64, 100)
    @time for i in 1:n
        state = getNextState(vec, base)
        vec = state[1]
        rad = state[2]
        radius[Int8(round(rad*50, RoundDown))+1] += 1
        if is_densitymatrix(make_densitymatrix(vec, base))
            n_tot += 1
            state = make_densitymatrix(vec, base)
            if PPT_criterion(state)
                n_ppt += 1
                radius_ppt[Int8(round(rad*50, RoundDown))+1] += 1
            end
            if majorization_criterion(state)
                n_maj += 1
            end
            #if reduction_criterion(state)
            #    n_red += 1
            #end
        end
    end
    return n_tot, n_ppt, n_maj, n_red, n_ppt/n, n_maj/n, n_red/n, radius, radius_ppt, radius_maj
end

function make_densitymatrix(x, base)
    result = 1/6 * Matrix(I,6,6)
    for i in 1:11
        result += x[i] * x_b6[:,:,i, base] 
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
    return (1 - p2 >= 0 - 1e-16)
end

function newton_3(p2,p3)
    return (1 - 3 * p2 + 2 * p3 >= 0 - 1e-16)
end

function newton_4(p2,p3,p4)
    return (1 - 6 * p2 + 8 * p3 + 3 * p2^2 - 6 * p4 >= 0 - 1e-16)
end

function newton_5(p2,p3,p4,p5)
    return (1 - 10 * p2 + 20 * p3- 30 * p4 + 24*p5 + 15 * p2^2 - 20*p2*p3  >= 0 - 1e-16 )
end

function newton_6(p2,p3,p4,p5,p6)
    return (1- 15 * p2 + 40 * p3 - 90 * p4 + 144*p5 - 120*p6
             + 45 * p2^2 - 120*p2*p3 + 90*p2*p4 + 40*p3^2 - 15*p2^3  >= 0 - 1e-16 )
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
    #=px = kron(ptrace(x, [2,3], 2), Matrix(I,3,3)) - x
    ev = real(eigvals(px))    
    #min - eigenwerte sind bereits sortiert
    return ev[1] > 0 +1e-14
    #return all(>=(0 + 1e-14), ev) =#
    #px = kron(Matrix(I,2,2),ptrace(x, [2,3], 1)) - x
    px = kron(Matrix(I,2,2),ptrace(x, [2,3], 1)) - x
    ev = real(eigvals(px))
    return ev[1] > 0 +1e-14
end


phi1 = 0
phi2 = 0
axis = 1



function final_run(n::Int, rep::Int)
    resPPT = zeros(rep)
    resMaj = zeros(rep)
    resBell = zeros(rep)
    radTot = zeros(rep,100)
    radPPT = zeros(rep,100)
    radMaj = zeros(rep,100) 
    radBell = zeros(rep,100) 
    for i in 1:rep
        resPPT[i], resMaj[i], resBell[i], radTot[i,:], radPPT[i,:], radMaj[i,:], radBell[i,:] = run(n)[[5,6,7,8,9,10,11]]
    end
    return resPPT, resMaj, resBell, radTot, radPPT, radMaj, radBell
end

res = final_run(100000,3)


function give_results(n,k)
    ppt = zeros((15,k))
    maj = zeros((15,k))
    for i in 1:15
        println(i)
        for j in 1:k
            res = run(n,i)[[5,6]]
            ppt[i,j] = res[1]
            maj[i,j] = res[2]
        end
    end
    println("PPT:")
    for i in 1:15
        print(i, " & ")
        for j in 1:k
            print(ppt[i,j], " & ")
        end
        print(mu_sigma(ppt[i,:])[1], " & ")
        print(mu_sigma(ppt[i,:])[2], " \\\\ \\hline")
        println("")
    end
    println("Maj:")
    for i in 1:15
        print(i, " & ")
        for j in 1:k
            print(maj[i,j], " & ")
        end
        print(mu_sigma(maj[i,:])[1], " & ")
        print(mu_sigma(maj[i,:])[2], " \\\\ \\hline")
        println("")
    end
    return ppt, maj
end
@time give_results(100,1)
@time res = give_results(500000,1)


