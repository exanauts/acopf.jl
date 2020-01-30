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
  cuPg = CuArray{Float64,1,Nothing}(zeros(Float64, nPg))
  cuQg = CuArray{Float64,1,Nothing}(zeros(Float64, nQg))
  cuVa = CuArray{Float64,1,Nothing}(zeros(Float64, nVa))
  cuVm = CuArray{Float64,1,Nothing}(zeros(Float64, nVm))
  curbalconst = CuArray{Float64,1,Nothing}(zeros(Float64, nbus))
  cuibalconst = CuArray{Float64,1,Nothing}(zeros(Float64, nbus))
  culimitsto = CuArray{Float64,1,Nothing}(zeros(Float64, nline))
  culimitsfrom = CuArray{Float64,1,Nothing}(zeros(Float64, nline))
  function eval_f(x::Vector{Float64})
    println("Returning objective")
    cuPg[:] = x[1:nPg] 
    cuQg[:] = x[nPg+1:nPg+nQg] 
    cuVa[:] = x[nPg+nQg+1:nPg+nQg+nVa] 
    cuVm[:] = x[nPg+nQg+nVa+1:end] 
    arrays = acopf.create_arrays(cuPg, cuQg, cuVa, cuVm, opfdata, Float64)
    println("Done returning objective")
    return acopf.objective(opfdata, arrays)
    # return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
  end

  function eval_g(x::Vector{Float64}, g::Vector{Float64})
    println("Returning gradient")
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
    println("Done returning gradient")
    # @show g
  end

  function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
    t1s{N} =  ForwardDiff.Dual{Nothing,Float64, N} where N
    t1sseedvec = zeros(Float64, ngen)
    t1sseeds = Array{ForwardDiff.Partials{ngen,Float64},1}(undef, ngen)
    for i in 1:ngen
      t1sseedvec[i] = 1.0
      t1sseeds[i] = ForwardDiff.Partials{ngen, Float64}(NTuple{ngen, Float64}(t1sseedvec))
      t1sseedvec[i] = 0.0
    end
    t1scuPg = acopf.myseed!(CuArray{t1s{ngen}, 1, Nothing}(undef, ngen), x[1:nPg], t1sseeds, timeroutput)
    t1scuQg = ForwardDiff.seed!(CuArray{t1s{ngen}, 1, Nothing}(undef, ngen), x[nPg+1:nPg+nQg])
    t1scuVa = ForwardDiff.seed!(CuArray{t1s{ngen}, 1, Nothing}(undef, nbus), x[nPg+nQg+1:nPg+nQg+nVa])
    t1scuVm = ForwardDiff.seed!(CuArray{t1s{ngen}, 1, Nothing}(undef, nbus), x[nPg+nQg+nVa+1:end])
    t1sarrays = acopf.create_arrays(t1scuPg, t1scuQg, t1scuVa, t1scuVm, opfdata, t1s{ngen})
    t1sgrad = acopf.objective(opfdata, t1sarrays)
    grad_f = ForwardDiff.partials.(t1sgrad).values
  end

  function eval_jac_g(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
    nconstraints = 2*nbus + 2*nline #number of constraints
    if mode == :Structure
      println("Returning Jacobian structure")
      idx = 1
      for c in 1:nconstraints #number of constraints
        for i in 1:n # number of variables
          rows[idx] = c ; cols[idx] = i
          idx += 1 
        end
      end
      println("Done returning Jacobian structure")
    else
      println("Returning Jacobian values")
      t1s{N} =  ForwardDiff.Dual{Nothing,Float64, N} where N
      t1sseedvec = zeros(Float64, n)
      t1sseeds = Array{ForwardDiff.Partials{n,Float64},1}(undef, n)
      for i in 1:n
        t1sseedvec[i] = 1.0
        t1sseeds[i] = ForwardDiff.Partials{n, Float64}(NTuple{n, Float64}(t1sseedvec))
        t1sseedvec[i] = 0.0
      end
      t1scuPg = acopf.myseed!(CuArray{t1s{n}, 1, Nothing}(undef, ngen), x[1:nPg], t1sseeds[1:nPg], timeroutput)
      t1scuQg = acopf.myseed!(CuArray{t1s{n}, 1, Nothing}(undef, ngen), x[nPg+1:nPg+nQg], t1sseeds[nPg+1:nPg+nQg], timeroutput)
      t1scuVa = acopf.myseed!(CuArray{t1s{n}, 1, Nothing}(undef, nbus), x[nPg+nQg+1:nPg+nQg+nVa], t1sseeds[nPg+nQg+1:nPg+nQg+nVa], timeroutput)
      t1scuVm = acopf.myseed!(CuArray{t1s{n}, 1, Nothing}(undef, nbus), x[nPg+nQg+nVa+1:end], t1sseeds[nPg+nQg+nVa+1:end], timeroutput)
      T = typeof(t1scuQg)
      t1srbalconst = T(undef, length(opfdata.buses))
      t1sibalconst = T(undef, length(opfdata.buses))
      t1slimitsto = T(undef, length(opfdata.lines))
      t1slimitsfrom = T(undef, length(opfdata.lines))
      t1sarrays = acopf.create_arrays(t1scuPg, t1scuQg, t1scuVa, t1scuVm, opfdata, t1s{n})
      acopf.constraints(t1srbalconst, t1sibalconst, t1slimitsto, t1slimitsfrom, opfdata, t1sarrays, timeroutput)
      for i in 1:nbus
        values[(i-1) * n + 1 : i*n] .= ForwardDiff.partials.(t1srbalconst[i]).values
      end
      for i in 1:nbus
        values[(i-1) * n + nbus + 1 : i*n + nbus] .= ForwardDiff.partials.(t1sibalconst[i]).values
      end
      for i in 1:nline
        values[(i-1) * n + 2*nbus + 1 : i*n + 2*nbus] .= ForwardDiff.partials.(t1slimitsto[i]).values
      end
      for i in 1:nline
        values[(i-1) * n + 2*nbus + nline + 1 : i*n + 2*nbus + nline] .= ForwardDiff.partials.(t1slimitsfrom[i]).values
      end
      println("Done returning Jacobian values")
    end
  end

  function eval_h(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Vector{Float64})
    if mode == :Structure
      println("Returning Hessian structure")
      idx = 1
      for row = 1:n
        for col = 1:n
          rows[idx] = row
          cols[idx] = col
          idx += 1
        end
      end
      println("Done returning Hessian structure")
    else
      println("Returning Hessian values")
      t1s{N} =  ForwardDiff.Dual{Nothing,Float64, N} where N
      t1sseedvec = zeros(Float64, n)
      t1sseeds = Array{ForwardDiff.Partials{n,Float64},1}(undef, n)
      for i in 1:n
        t1sseedvec[i] = 1.0
        t1sseeds[i] = ForwardDiff.Partials{n, Float64}(NTuple{n, Float64}(t1sseedvec))
        t1sseedvec[i] = 0.0
      end
      t1scuPg = acopf.myseed!(CuArray{t1s{n}, 1, Nothing}(undef, ngen), x[1:nPg], t1sseeds[1:nPg], timeroutput)
      t1scuQg = acopf.myseed!(CuArray{t1s{n}, 1, Nothing}(undef, ngen), x[nPg+1:nPg+nQg], t1sseeds[nPg+1:nPg+nQg], timeroutput)
      t1scuVa = acopf.myseed!(CuArray{t1s{n}, 1, Nothing}(undef, nbus), x[nPg+nQg+1:nPg+nQg+nVa], t1sseeds[nPg+nQg+1:nPg+nQg+nVa], timeroutput)
      t1scuVm = acopf.myseed!(CuArray{t1s{n}, 1, Nothing}(undef, nbus), x[nPg+nQg+nVa+1:end], t1sseeds[nPg+nQg+nVa+1:end], timeroutput)

      t2s{M,N} =  ForwardDiff.Dual{Nothing,t1s{N}, M} where {N, M}
      t2sseedvec = Array{t1s{n},1}(undef, n)
      t2sseeds = Array{ForwardDiff.Partials{n,t1s{n}},1}(undef, n)
      t2sseedvec .= 0.0
      for i in 1:n
        t2sseedvec[i] = 1.0
        t2sseeds[i] = ForwardDiff.Partials{n, t1s{n}}(NTuple{n, t1s{n}}(t2sseedvec))
        t2sseedvec[i] = 0.0
      end
      println("Allocating t2s")
      # t2scuPg = acopf.myseed!(CuArray{t2s{n,n}, 1, Nothing}(undef, ngen), t1scuPg, t2sseeds[1:nPg], timeroutput) 
      # t2scuQg = acopf.myseed!(CuArray{t2s{n,n}, 1, Nothing}(undef, ngen), t1scuQg, t2sseeds[nPg+1:nPg+nQg], timeroutput)
      # t2scuVa = acopf.myseed!(CuArray{t2s{n,n}, 1, Nothing}(undef, nbus), t1scuVa, t2sseeds[nPg+nQg+1:nPg+nQg+nVa], timeroutput)
      # t2scuVm = acopf.myseed!(CuArray{t2s{n,n}, 1, Nothing}(undef, nbus), t1scuVm, t2sseeds[nPg+nQg+nVa+1:end], timeroutput) 
      t2scuPg = ForwardDiff.seed!(CuArray{t2s{n,n}, 1, Nothing}(undef, ngen), t1scuPg)
      t2scuQg = ForwardDiff.seed!(CuArray{t2s{n,n}, 1, Nothing}(undef, ngen), t1scuQg)
      t2scuVa = ForwardDiff.seed!(CuArray{t2s{n,n}, 1, Nothing}(undef, nbus), t1scuVa)
      t2scuVm = ForwardDiff.seed!(CuArray{t2s{n,n}, 1, Nothing}(undef, nbus), t1scuVm)
      println("Done allocating t2s")
      T = typeof(t2scuQg)
      t2srbalconst = T(undef, length(opfdata.buses))
      t2sibalconst = T(undef, length(opfdata.buses))
      t2slimitsto = T(undef, length(opfdata.lines))
      t2slimitsfrom = T(undef, length(opfdata.lines))
      println("Create arrays")
      t2sarrays = acopf.create_arrays(t2scuPg, t2scuQg, t2scuVa, t2scuVm, opfdata, t2s{n,n})
      println("objective")
      t2sPg = acopf.objective(opfdata, t2sarrays)
      @show ForwardDiff.partials.(ForwardDiff.partials(t2scuPg[1]).values)
      @show ForwardDiff.partials(t2scuPg[1].value)
      # ForwardDiff.partials(t2scuPg[1])
      @show typeof(t2sPg)
      @show typeof(t2scuPg)
      return
      println("constraints")
      acopf.constraints(t2srbalconst, t2sibalconst, t2slimitsto, t2slimitsfrom, opfdata, t2sarrays, timeroutput)
      println("print")
      for (i, row) in enumerate(ForwardDiff.partials.(ForwardDiff.partials.(t2sPg).values))
        for (j, col) in enumerate(row.values)
          @show i, j
        end
      end
      println("Done returning Hessian values")
    end
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
  # function intermediate(alg_mod::Int, iter_count::Int,
  # obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
  # d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
  # ls_trials::Int)
  # return iter_count < 1  # Interrupts after one iteration.
  # end

  # setIntermediateCallback(prob, intermediate)

  solvestat = solveProblem(prob)

  # @test Ipopt.ApplicationReturnStatus[solvestat] == :User_Requested_Stop

end
end