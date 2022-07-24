using LinearAlgebra
using Random
using QuantumInformation

#Initializing base matrices:
#Basis d=2
b2 = Array{Complex{Float64}}(undef, 2, 2, 4)

b2[:,:,1] = [1 0; 0 1]/sqrt(2)
b2[:,:,2] = [0 1; 1 0]/sqrt(2)
b2[:,:,3] = [0 -1im; 1im 0]/sqrt(2)
b2[:,:,4] = [1 0; 0 -1]/sqrt(2)

k16 = Array{Complex{Float64}}(undef, 4, 4, 16)
for i in 0:3, j in 1:4
    k16[:,:,4*i+j] = kron(b2[:,:,i+1], b2[:,:,j])
end

k15 = k16[:,:,2:16]


function commutator(x,y)
    return x*y - y*x
end

#Basis X-states 2x2
x_b4 = Array{Complex{Float64}}(undef, 4, 4, 7, 15)

#Commuting element: tau_x
comm = k16[:,:,2] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,1] = k16[:,:,i]
    end
end
#Commuting element: tau_y
comm = k16[:,:,3] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,2] = k16[:,:,i]
    end
end

for i in 1:7
    display(x_b4[:,:,i,2])
end
#Commuting element: tau_z


comm = k16[:,:,4] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,3] = k16[:,:,i]
    end
end
#Commuting element: sigma_x
comm = k16[:,:,5] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,4] = k16[:,:,i]
    end
end
#Commuting element: sigma_x tau_x
x_b4[:,:,1,5] = kron(b2[:,:,2], b2[:,:,2])
x_b4[:,:,2,5] = kron(b2[:,:,4], b2[:,:,3])
x_b4[:,:,3,5] = kron(b2[:,:,1], b2[:,:,2])
x_b4[:,:,4,5] = kron(b2[:,:,4], b2[:,:,4])
x_b4[:,:,5,5] = kron(b2[:,:,3], b2[:,:,4])
x_b4[:,:,6,5] = kron(b2[:,:,2], b2[:,:,1])
x_b4[:,:,7,5] = kron(b2[:,:,3], b2[:,:,3])
#Commuting element: sigma_x tau_y
comm = k16[:,:,7] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,6] = k16[:,:,i]
    end
end
#Commuting element: sigma_x tau_z
comm = k16[:,:,8] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,7] = k16[:,:,i]
    end
end
#Commuting element: sigma_y
comm = k16[:,:,9] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,8] = k16[:,:,i]
    end
end
#Commuting element: sigma_y tau_x
comm = k16[:,:,10] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,9] = k16[:,:,i]
    end
end
#Commuting element: sigma_y tau_y
comm = k16[:,:,11] 

index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,10] = k16[:,:,i]
    end
end
#Commuting element: sigma_y tau_z
comm = k16[:,:,12] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,11] = k16[:,:,i]
    end
end
#Commuting element: sigma_z
x_b4[:,:,1,12] = kron(b2[:,:,4], b2[:,:,1])
x_b4[:,:,2,12] = kron(b2[:,:,1], b2[:,:,2])
x_b4[:,:,3,12] = kron(b2[:,:,1], b2[:,:,3])
x_b4[:,:,4,12] = kron(b2[:,:,1], b2[:,:,4])
x_b4[:,:,5,12] = kron(b2[:,:,4], b2[:,:,2])
x_b4[:,:,6,12] = kron(b2[:,:,4], b2[:,:,3])
x_b4[:,:,7,12] = kron(b2[:,:,4], b2[:,:,4])
#Commuting element: sigma_z tau_x
comm = k16[:,:,14] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,13] = k16[:,:,i]
    end
end
#Commuting element: sigma_z tau_y
comm = k16[:,:,15] 
index = 0
for i in 2:16
    if norm(commutator(comm, k16[:,:,i]) - zeros(4,4)) < 1e-15
        index += 1
        x_b4[:,:,index,14] = k16[:,:,i]
    end
end
#Commuting element: sigma_z tau_z
x_b4[:,:,1,15] = kron(b2[:,:,1], b2[:,:,4])
x_b4[:,:,2,15] = kron(b2[:,:,4], b2[:,:,1])
x_b4[:,:,3,15] = kron(b2[:,:,2], b2[:,:,2])
x_b4[:,:,4,15] = kron(b2[:,:,2], b2[:,:,3])
x_b4[:,:,5,15] = kron(b2[:,:,3], b2[:,:,2])
x_b4[:,:,6,15] = kron(b2[:,:,3], b2[:,:,3])
x_b4[:,:,7,15] = kron(b2[:,:,4], b2[:,:,4])



function getRandomUnitVector(n)
    vec = real(randn(n))
    while norm(vec) < 0.0001
        vec = real(randn(n))
    end
    return vec/norm(vec)
end

function rotation(phi)
    return exp(sqrt(2) * (phi[1]*b2[:,:,2] + phi[2]*b2[:,:,3] + phi[3]*b2[:,:,4]))
end

function getNextState(state, base)
    dir = getRandomUnitVector(7)  
    vmax =  2 * sqrt(3 / 4)
    vmin = -2 * sqrt(3 / 4)
    v = vmin + (vmax-vmin)*rand()
    a = state + v * dir
    while !is_densitymatrix(make_densitymatrix(a, base))
        (v < 0) ? (vmin = v) : (vmax = v)
        v = vmin + (vmax - vmin) * rand()
        a = state + v * dir
    end 
    return a
end

function Base.run(n::Int, base::Int)
    n_ppt = 0
    n_maj = 0
    n_red = 0
    vec = zeros(Float64, 7)
    @time Threads.@threads for i in 1:n
        vec = getNextState(vec, base)
        state = make_densitymatrix(vec, base)
        if PPT_criterion(state)
            n_ppt += 1
        end
        if majorization_criterion(state)
            n_maj += 1
        end
        if reduction_criterion(state)
            n_red += 1
        end
    end
    return [n n_ppt n_maj n_red n_ppt/n n_maj/n n_red/n]
end

function make_densitymatrix(x, base)
    result = 1/4 * Matrix(I,4,4)
    for i in 1:7
        result += x[i] * x_b4[:,:,i,base] 
    end
    return kron(rotation(phi1),rotation(phi2)) * result * inv(kron(rotation(phi1), rotation(phi2)))
end

function matrixpower(x, n)
    if n==1 || n==0
        return x
    else
        return x * matrixpower(x, n-1)
    end
end

function is_densitymatrix(x)
    if abs(real(tr(x))-1) > 1e-10
        throw(ArgumentError("Trace not equal one!"))
    end
    if abs(maximum(real(transpose(conj(x)) - x))) > 1e-14 || abs(maximum(real(x - transpose(conj(x))))) > 1e-14 
        throw(ArgumentError("Matrix is not hermitian!"))
    end 
    p2 = real(tr(matrixpower(x,2)))
    p3 = real(tr(matrixpower(x,3)))
    p4 = real(tr(matrixpower(x,4)))
    if newton_2(p2) && newton_3(p2, p3) && newton_4(p2, p3, p4)
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

function x_flip4(a)
    return [a[1] a[2] a[3] -a[4] -a[5] a[6] a[7]]
end

broadcast(x -> (abs(x) < 1e-15) ? ComplexF64(0) : x , partialtranspose(make_densitymatrix([0.1,0.2,0.1,0.1,0.2,-0.1,0.2],15),2,[2,2]))
broadcast(x -> (abs(x) < 1e-15) ? ComplexF64(0) : x , make_densitymatrix(x_flip4([0.1,0.2,0.1,0.1,0.2,-0.1,0.2]),15))


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
    return reshape(permutedims(r,p),(d,d))
end

function PPT_criterion(x)
    return is_densitymatrix(partialtranspose(x, 2, [2,2]))
end

function majorization_criterion(x)
    return (maj_eigenvalues(x, ptrace(x,[2,2],1)) && maj_eigenvalues(x, ptrace(x,[2,2],2)))
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
    return is_densitymatrix(kron(ptrace(x, [2,2], 2),  Matrix(I,2,2)) - x)
end


function final_run(n::Int, rep::Int)
    res1 = zeros(rep,15)
    res2 = zeros(rep,15)
    res3 = zeros(rep,15)
    for j in 1:15
        for i in 1:rep
            res1[i,j], res2[i,j], res3[i,j] = run(n,j)[[5,6,7]]
        end
    end
    return res1, res2, res3
    #return sum(res1)/length(res1), sum(res2)/length(res2), sum(res3)/length(res3)
end

@time res = final_run(1000000,3)


function mu_sigma(arr)
    mean = sum(arr)/length(arr)
    var = 0
    for i in 1:length(arr)
        var = (mean - arr[i])^2
    end
    var = var/(length(arr)-1)
    sig = sqrt(var)
    return mean,sig
end











