module IpoptTest
include("acopf.jl")
using Ipopt
using Test
using .acopf
using ForwardDiff
using CuArrays, CUDAnative

# hs071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)


function test(Pg0, Qg0, Vm0, Va0, npartials, mpartials, timeroutput, case)
  opfdata = acopf.opf_loaddata(case)
  t1s{N} =  ForwardDiff.Dual{Nothing,Float64, N} where N
  t2s{M,N} =  ForwardDiff.Dual{Nothing,t1s{N}, M} where {N, M}
  nPg = size(Pg0,1) ; nQg = size(Qg0,1) ; nVm = size(Vm0,1) ; nVa = size(Va0,1)
  n = nPg + nQg + nVm + nVa
  nbus = length(opfdata.buses)
  nline = length(opfdata.lines)
  ngen = length(opfdata.generators)
  m = 2 * nbus + 2 * nline
  # m = 0 
  cuPg = CuArray{Float64,1,Nothing}(zeros(Float64, nPg))
  cuQg = CuArray{Float64,1,Nothing}(zeros(Float64, nQg))
  cuVa = CuArray{Float64,1,Nothing}(zeros(Float64, nVa))
  cuVm = CuArray{Float64,1,Nothing}(zeros(Float64, nVm))
  curbalconst = CuArray{Float64,1,Nothing}(zeros(Float64, nbus))
  cuibalconst = CuArray{Float64,1,Nothing}(zeros(Float64, nbus))
  culimitsto = CuArray{Float64,1,Nothing}(zeros(Float64, nline))
  culimitsfrom = CuArray{Float64,1,Nothing}(zeros(Float64, nline))
  function eval_f(x::Vector{Float64})
    cuPg[:] = x[1:nPg] 
    cuQg[:] = x[nPg+1:nPg+nQg] 
    cuVa[:] = x[nPg+nQg+1:nPg+nQg+nVa] 
    cuVm[:] = x[nPg+nQg+nVa+1:end] 
    arrays = acopf.create_arrays(cuPg, cuQg, cuVa, cuVm, opfdata, Float64)
    return acopf.objective(opfdata, arrays)
    # return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
  end

  function eval_g(x::Vector{Float64}, g::Vector{Float64})
    cuPg[:] = x[1:nPg] 
    cuQg[:] = x[nPg+1:nPg+nQg] 
    cuVa[:] = x[nPg+nQg+1:nPg+nQg+nVa] 
    cuVm[:] = x[nPg+nQg+nVa+1:end] 
    arrays = acopf.create_arrays(cuPg, cuQg, cuVa, cuVm, opfdata, Float64)
    # @show x
    # @show cuVm
    acopf.constraints(curbalconst, cuibalconst, culimitsto, culimitsfrom, opfdata, arrays, timeroutput)
    g[1:nbus] = curbalconst[:]
    g[nbus+1:2*nbus] = cuibalconst[:]
    g[2*nbus+1:2*nbus+nline] = culimitsto[:]
    g[2*nbus+nline+1:end] = culimitsfrom[:]
    # @show g
  end

  function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
    function objective(x)
      Pg = x[1:nPg]
      Qg = x[nPg+1:nPg+nQg]
      Va = x[nPg+nQg+1:nPg+nQg+nVa]
      Vm = x[nPg+nQg+nVa+1:end]
      arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, typeof(x))
      return acopf.objective(opfdata, arrays)
    end
    cux = CuArray{Float64,1,Nothing}(x)
    g = cux -> ForwardDiff.gradient(objective, cux)
    grad_f[:] = g(cux)
    return
  end

  function eval_jac_g(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
    nconstraints = 2*nbus + 2*nline #number of constraints
    if mode == :Structure
      idx = 1
      for c in 1:nconstraints #number of constraints
        for i in 1:n # number of variables
          rows[idx] = c ; cols[idx] = i
          idx += 1 
        end
      end
    else
      function constraints(x)
        Pg = x[1:nPg]
        Qg = x[nPg+1:nPg+nQg]
        Va = x[nPg+nQg+1:nPg+nQg+nVa]
        Vm = x[nPg+nQg+nVa+1:end]
        arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, typeof(x))
        T = typeof(x)
        rbalconst   = T(undef, nbus)
        ibalconst   = T(undef, nbus)
        limitsto    = T(undef, nline)
        limitsfrom  = T(undef, nline)
        acopf.constraints(rbalconst, ibalconst, limitsto, limitsfrom, opfdata, arrays, timeroutput)
        y = T(undef, 2*nbus+2*nline)
        y[1:nbus] = rbalconst[:] 
        y[nbus+1:2*nbus] = ibalconst[:] 
        y[2*nbus+1:2*nbus+nline] = limitsto[:] 
        y[2*nbus+nline+1:end] = limitsfrom[:] 
        return y
      end
      cux = CuArray{Float64,1,Nothing}(x)
      fjac = cux -> ForwardDiff.jacobian(constraints, cux)
      jac = fjac(cux)
      k = 1
      for i in 1:size(jac,1)
        for j in 1:size(jac,2)
          values[k] = jac[i,j]
          k += 1
        end
      end
    end
    return
  end

  function eval_h(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Vector{Float64})
    if mode == :Structure
      idx = 1
      for row = 1:n
        for col = 1:n
          rows[idx] = row
          cols[idx] = col
          idx += 1
        end
      end
    else
      function objective(x)
        Pg = x[1:nPg]
        Qg = x[nPg+1:nPg+nQg]
        Va = x[nPg+nQg+1:nPg+nQg+nVa]
        Vm = x[nPg+nQg+nVa+1:end]
        arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, typeof(x))
        return acopf.objective(opfdata, arrays)
      end
      cux = CuArray{Float64,1,Nothing}(x)
      h = cux -> ForwardDiff.hessian(objective, cux)
      objhess = h(cux)
      k = 1
      for i in 1:size(x,1)
        for j in 1:size(x,1)
          values[k] = obj_factor * objhess[i,j] 
          k += 1
        end
      end
      select = 1
      function constraints(x)
        Pg = x[1:nPg]
        Qg = x[nPg+1:nPg+nQg]
        Va = x[nPg+nQg+1:nPg+nQg+nVa]
        Vm = x[nPg+nQg+nVa+1:end]
        arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, typeof(x))
        T = typeof(x)
        rbalconst   = T(undef, nbus)
        ibalconst   = T(undef, nbus)
        limitsto    = T(undef, nline)
        limitsfrom  = T(undef, nline)
        acopf.constraints(rbalconst, ibalconst, limitsto, limitsfrom, opfdata, arrays, timeroutput)
        y = T(undef, 2*nbus+2*nline)
        y[1:nbus] = rbalconst[:] 
        y[nbus+1:2*nbus] = ibalconst[:] 
        y[2*nbus+1:2*nbus+nline] = limitsto[:] 
        y[2*nbus+nline+1:end] = limitsfrom[:] 
        return y[select]
      end
      chess = cux -> ForwardDiff.hessian(constraints, cux)
      for l in 1:m
        select = l
        conshess = chess(cux)
        k = 1
        for i in 1:size(x,1)
          for j in 1:size(x,1)
            values[k] += lambda[l] * conshess[i,j]
            k += 1
          end
        end
      end
    end
    return
  end
  x_L = Vector{Float64}(undef,n)
  x_U = Vector{Float64}(undef,n)
  for (i,g) in enumerate(opfdata.generators)
    x_L[i] = g.Pmin
    x_U[i] = g.Pmax
    x_L[nPg+i] = g.Qmin
    x_U[nPg+i] = g.Qmax
  end
  for (i,b) in enumerate(opfdata.buses)
    x_L[nPg + nQg + i] = b.Vmin
    x_U[nPg + nQg + i] = b.Vmax
    x_L[nPg + nQg + nVa + i] = -2.0e19
    x_U[nPg + nQg + nVa + i] = 2.0e19
  end

  g_L = Vector{Float64}(undef, m)
  g_U = Vector{Float64}(undef, m)
  for i in 1:2*nbus g_L[i] = 0.0 end
  for i in 1:2*nbus g_U[i] = 0.0 end
  for i in 2*nbus+1:m g_L[i] = -2.0e19 end
  for i in 2*nbus+1:m g_U[i] = 0.0 end

  prob = createProblem(n, x_L, x_U, m, g_L, g_U, n*m, n*n,
                      eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
  @show Pg0
  @show Qg0
  @show Vm0
  @show Va0
  prob.x[1:nPg] = Pg0[:] 
  prob.x[nPg+1:nPg+nQg] = Qg0[:]
  prob.x[nPg+nQg+1:nPg+nQg+nVa] = Va0[:] 
  prob.x[nPg+nQg+nVa+1:end] = Vm0[:]

  @show prob.x
  @show eval_f(prob.x)
  # g = Vector{Float64}(undef, 2*nbus+2*nline)
  g = zeros(Float64, 2*nbus+2*nline)
  eval_g(prob.x, g)
  @show g

  @show g_L
  @show g_U
  @show x_L
  @show x_U
  
  # # This tests callbacks.
  function intermediate(alg_mod::Int, iter_count::Int,
  obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
  d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
  ls_trials::Int)
  return iter_count < 50  # Interrupts after one iteration.
  end

  setIntermediateCallback(prob, intermediate)

  solvestat = solveProblem(prob)

  # @test Ipopt.ApplicationReturnStatus[solvestat] == :User_Requested_Stop
  return

end
end