using LinearAlgebra
using Random
using QuantumInformation
using Optim
using Plots
using JuMP


#Basis d=2
b2 = Array{Complex{Float64}}(undef, 2, 2, 4)

b2[:,:,1] = [1 0; 0 1]/sqrt(2)
b2[:,:,2] = [0 1; 1 0]/sqrt(2)
b2[:,:,3] = [0 -1im; 1im 0]/sqrt(2)
b2[:,:,4] = [1 0; 0 -1]/sqrt(2)

k16 = Array{Complex{Float64}}(undef, 4, 4, 16)
for i in 0:3
    for j in 1:4
        k16[:,:,4*i+j] = kron(b2[:,:,i+1],b2[:,:,j])
    end
end
k15 = k16[:,:,2:16]

#Basis X-states 2x2
x_b4 = Array{Complex{Float64}}(undef, 4, 4, 7)
x_b4[:,:,1] = kron(b2[:,:,1], b2[:,:,4])
x_b4[:,:,2] = kron(b2[:,:,4], b2[:,:,1])
x_b4[:,:,3] = kron(b2[:,:,2], b2[:,:,2])
x_b4[:,:,4] = kron(b2[:,:,2], b2[:,:,3])
x_b4[:,:,5] = kron(b2[:,:,3], b2[:,:,2])
x_b4[:,:,6] = -1 * kron(b2[:,:,3], b2[:,:,3])
x_b4[:,:,7] = kron(b2[:,:,4], b2[:,:,4])

function getRandomUnitVector(n)
    vec = real(randn(n))
    while norm(vec) < 0.0001
        vec = real(randn(n))
    end
    return vec/norm(vec)
end

function rotation(phi)
    [cos(phi) -sin(phi) ; sin(phi) cos(phi)]
end

function getNextState(state)
    dir = getRandomUnitVector(7) 
    vmax =  2 * sqrt(3 / 4) 
    vmin = -2 * sqrt(3 / 4) 
    v = vmin + (vmax-vmin)*rand()
    a = state + v * dir
    while !is_densitymatrix(make_densitymatrix(a))
        (v < 0) ? (vmin = v) : (vmax = v)
        v = vmin + (vmax - vmin) * rand()
        a = state + v * dir
    end 
    rad = norm(a)/sqrt(3/4)
    return a, rad
end


function Base.run(n::Int)
    n_ppt = 0
    n_maj = 0
    n_bell = 0 #amount of states which violate the Bell inequality
    n_CG = 0
    n_genBell = 0
    radius = zeros(Int64, 100)
    radius_ppt = zeros(Int64, 100)
    radius_maj = zeros(Int64, 100)
    radius_bell = zeros(Int64, 100)
    radius_CG = zeros(Int64,100)
    radius_genBell = zeros(Int64,100)
    vec = zeros(Float64, 7) 
    @time Threads.@threads for i in 1:n
        state = getNextState(vec)
        vec = state[1]
        rad = state[2]
        
        state = make_densitymatrix(vec)
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
                
            end
            if violates_bell(state)
                n_bell += 1
                radius_bell[Int8(round(rad*50, RoundDown))+1] += 1
            end
            if violates_CG(state)
                n_CG += 1
                radius_CG[Int8(round(rad*50, RoundDown))+1] += 1
            end
            #=if general_Bell(state)
                n_genBell += 1
                radius_genBell[Int8(round(rad*50, RoundDown))+1] += 1
            end=#
            if majorization_criterion(state)
                n_maj += 1
                radius_maj[Int8(round(rad*50, RoundDown))+1] += 1
            end
        end
    end
    return n_ppt/n, n_maj/n, n_bell/n, n_CG/n, #=n_genBell/n,=# radius, radius_ppt, radius_maj, radius_bell, radius_CG#, radius_genBell
end



function sphere_vector(phi, theta)
    return [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]
end

function dagger(x)
    return transpose(conj(x))
end

function violates_bell(rho)
    C = getCoordMatrix(rho)
    ev = eigvals(dagger(C) * C)
    ev_sort = sort(ev, rev = true)
    return ev_sort[1] + ev_sort[2] > 1 + 1e-14
end

function violates_CG(rho)
    tauA = real(getVecA(rho))
    tauB = real(getVecB(rho))
    Crho = transpose(real(getCoordMatrix(rho)))
    ineq(x) = 4 + dot(sphere_vector(x[1],x[2])+sphere_vector(x[3],x[4]),tauB) - norm(Crho * (sphere_vector(x[1],x[2])+sphere_vector(x[3],x[4])+sphere_vector(x[5],x[6])) + tauA) - norm(Crho * (sphere_vector(x[1],x[2])+sphere_vector(x[3],x[4])-sphere_vector(x[5],x[6])) + tauA) - norm(Crho * (sphere_vector(x[1],x[2])-sphere_vector(x[3],x[4])))
    result = give_minimum(ineq)
    return result <= 1e-15
end

function give_minimum(f)
    res = optimize(f,ones(6)*pi/2)
    Optim.minimum(res)
end

function general_Bell(rho)
    tauA = real(getVecA(rho))
    tauB = real(getVecB(rho))
    Crho = transpose(real(getCoordMatrix(rho)))
    function measurement(l1a,l1b,l2a,l2b,phia,thetaa,phib,thetab)
        l1a * l1b + l1a * l2b * dot(sphere_vector(phib, thetab),tauB) + l2a * l1b * dot(sphere_vector(phia,thetaa),tauA) + l2a * l2b * dot(sphere_vector(phia,thetaa), Crho * sphere_vector(phib,thetab))
    end
    function measurementa(l1,l2,phi,theta)
        l1 + l2 * dot(sphere_vector(phi, theta),tauA)
    end
    function measurementb(l1,l2,phi,theta)
        l1 + l2 * dot(sphere_vector(phi, theta),tauB)
    end
    function meanmeas(l1a,l1b,l2a,l2b,phia,thetaa,phib,thetab)
        4*l1a*l1b - 2*(l1a+l1b) + 1 + (4*l1a*l2b-2*l2b)*dot(sphere_vector(phib, thetab),tauB) + (4*l2a*l1b-2*l2a)*dot(sphere_vector(phia, thetaa),tauA) + 4*l2a*l2b*dot(sphere_vector(phia, thetaa),Crho * sphere_vector(phib, thetab))
    end
    #=function ineq(l1a,l1a_,l1b,l1b_,l2a,l2a_,l2b,l2b_,phia,phia_,phib,phib_,thetaa,thetaa_,thetab,thetab_)
        (2*measurement(l1a,l1b,l2a,l2b,phia,thetaa,phib,thetab) - 1) - (2*measurement(l1a,l1b_,l2a,l2b_,phia,thetaa,phib_,thetab_)-1) + (2*measurement(l1a_,l1b,l2a_,l2b,phia_,thetaa_,phib,thetab)-1) + (2*measurement(l1a_,l1b_,l2a_,l2b_,phia_,thetaa_,phib_,thetab_)-1)
    end=#
    function ineq(l1a,l1a_,l1b,l1b_,l2a,l2a_,l2b,l2b_,phia,phia_,phib,phib_,thetaa,thetaa_,thetab,thetab_)
        meanmeas(l1a,l1b,l2a,l2b,phia,thetaa,phib,thetab) - meanmeas(l1a,l1b_,l2a,l2b_,phia,thetaa,phib_,thetab_)+meanmeas(l1a_,l1b,l2a_,l2b,phia_,thetaa_,phib,thetab)+meanmeas(l1a_,l1b_,l2a_,l2b_,phia_,thetaa_,phib_,thetab_)
    end
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_cpu_time", 60.0)
    #set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "tol", 1e-1)
    #model = Model(NLopt.Optimizer)
    @variable(model, 0<= l1a <= 1,start=0)
    @variable(model, 0<= l1a_ <= 1,start=0)
    @variable(model, 0<= l1b <= 1,start=0)
    @variable(model, 0<= l1b_ <= 1,start=0)
    @variable(model, 0<= l2a <= 1,start=1)
    @variable(model, 0<= l2a_ <= 1,start=1)
    @variable(model, 0<= l2b <= 1,start=1)
    @variable(model, 0<= l2b_ <= 1,start=1)
    @variable(model, 0<= phia <= 2*pi,start=0)
    @variable(model, 0<= phia_ <= 2*pi,start=0)
    @variable(model, 0<= phib <= 2*pi,start=0)
    @variable(model, 0<= phib_ <= 2*pi,start=0)
    @variable(model, 0<= thetaa <= 2*pi,start=0)
    @variable(model, 0<= thetaa_ <= 2*pi,start=pi/2)
    @variable(model, 0<= thetab <= 2*pi,start=pi/4)
    @variable(model, 0<= thetab_ <= 2*pi,start=-pi/4)
    register(model, :ineq, 16, ineq; autodiff = true)
    register(model, :min, 2, min; autodiff = true)
    @NLobjective(model, Max, ineq(l1a,l1a_,l1b,l1b_,l2a,l2a_,l2b,l2b_,phia,phia_,phib,phib_,thetaa,thetaa_,thetab,thetab_))
    @NLconstraint(model, l2a<=min(l1a,1-l1a))
    @NLconstraint(model, l2a_<=min(l1a_,1-l1a_))
    @NLconstraint(model, l2b<=min(l1b,1-l1b))
    @NLconstraint(model, l2b_<=min(l1b_,1-l1b_))
    optimize!(model)
    value.(l2a_)
    objective_value(model)
    #@show termination_status(model)
    #@show primal_status(model)
    #@show dual_status(model)
    #@show objective_value(model)
    #https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/
    return objective_value(model) > 2 + 1e-4
end


function getCoordMatrix(rho)
    C = zeros(Complex{Float64},(3,3))
    for i in 1:3
        for j in 1:3
            C[i,j] = 2*tr(rho * kron(b2[:,:,i+1], b2[:,:,j+1]))
        end
    end
    return C
end

function getVecA(rho)
    C = zeros(Complex{Float64},3)
    for i in 1:3    
        C[i] = 2*tr(rho * kron(b2[:,:,i+1], b2[:,:,1]))
    end
    return C
end

function getVecB(rho)
    C = zeros(Complex{Float64},3)
    for i in 1:3    
        C[i] = 2*tr(rho * kron(b2[:,:,1], b2[:,:,i+1]))
    end
    return C
end
function make_densitymatrix(x)
    result = 1/4 * Matrix(I,4,4)
    for i in 1:7
        result += x[i] * x_b4[:,:,i] 
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
    return (1 - p2 >= 0 + 1e-16)
end

function newton_3(p2,p3)
    return (1 - 3 * p2 + 2 * p3 >= 0 + 1e-16)
end

function newton_4(p2,p3,p4)
    return (1 - 6 * p2 + 8 * p3 + 3 * p2^2 - 6 * p4 >= 0 + 1e-16)
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

final_run(1000,2)
@time res = final_run(5000000, 4)

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



