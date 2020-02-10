module IpoptTest
include("acopf.jl")
using Ipopt
using Test
using .acopf
using ForwardDiff
using CuArrays, CUDAnative
using TimerOutputs

# hs071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)


function test(Pg0, Qg0, Vm0, Va0, npartials, mpartials, timeroutput, case; max_iter=100)
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
  io = open("test.out", "w+")
  function myprint(name,var)
      return
      write(io, name * ": ") 
      for v in var
        if v != 0.0
          write(io, string(v) * " ")
        end
      end
      write(io,"\n")
  end 
  function eval_f(x::Vector{Float64})
    @timeit timeroutput "eval_f" begin
    cuPg[:] = x[1:nPg] 
    cuQg[:] = x[nPg+1:nPg+nQg] 
    cuVa[:] = x[nPg+nQg+1:nPg+nQg+nVa] 
    cuVm[:] = x[nPg+nQg+nVa+1:end] 
    arrays = acopf.create_arrays(cuPg, cuQg, cuVa, cuVm, opfdata, timeroutput)
    obj = acopf.objective(opfdata, arrays, timeroutput)
    myprint("fx", x)
    myprint("fy", obj)
    end
    return obj
  end

  function eval_g(x::Vector{Float64}, g::Vector{Float64})
    @timeit timeroutput "eval_g" begin
    cuPg[:] = x[1:nPg] 
    cuQg[:] = x[nPg+1:nPg+nQg] 
    cuVa[:] = x[nPg+nQg+1:nPg+nQg+nVa] 
    cuVm[:] = x[nPg+nQg+nVa+1:end] 
    arrays = acopf.create_arrays(cuPg, cuQg, cuVa, cuVm, opfdata, timeroutput)
    acopf.constraints(curbalconst, cuibalconst, culimitsto, culimitsfrom, opfdata, arrays, timeroutput)
    g[1:nbus] = curbalconst[:]
    g[nbus+1:2*nbus] = cuibalconst[:]
    g[2*nbus+1:2*nbus+nline] = culimitsto[:]
    g[2*nbus+nline+1:end] = culimitsfrom[:]
    myprint("gx", x)
    myprint("g", g)
    end
  end

  function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
    @timeit timeroutput "eval_grad_f" begin
    function objective(x)
      Pg = x[1:nPg]
      Qg = x[nPg+1:nPg+nQg]
      Va = x[nPg+nQg+1:nPg+nQg+nVa]
      Vm = x[nPg+nQg+nVa+1:end]
      arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, timeroutput)
      return acopf.objective(opfdata, arrays, timeroutput)
    end
    cux = CuArray{Float64,1,Nothing}(x)
    g = cux -> ForwardDiff.gradient(objective, cux)
    grad_f[:] = g(cux)
    myprint("grad_x", x)
    myprint("grad_f", grad_f)
    end
    return
  end

  function eval_jac_g(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
    @timeit timeroutput "eval_jac_g" begin
    if mode == :Structure
      idx = 1
      for c in 1:m #number of constraints
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
        acopf.update_arrays!(arrays, Pg, Qg, Va, Vm, opfdata, timeroutput)
        acopf.constraints(rbalconst, ibalconst, limitsto, limitsfrom, opfdata, arrays, timeroutput)
        y[1:nbus] = rbalconst[:] 
        y[nbus+1:2*nbus] = ibalconst[:] 
        y[2*nbus+1:2*nbus+nline] = limitsto[:] 
        y[2*nbus+nline+1:end] = limitsfrom[:] 
        return y
      end
      cux = CuArray{Float64,1,Nothing}(x)
      cfg = ForwardDiff.JacobianConfig(constraints, cux)
      rbalconst   = CuArray{eltype(cfg)}(undef, nbus)
      ibalconst   = CuArray{eltype(cfg)}(undef, nbus)
      limitsto    = CuArray{eltype(cfg)}(undef, nline)
      limitsfrom  = CuArray{eltype(cfg)}(undef, nline)
      y = CuArray{eltype(cfg)}(undef, 2*nbus+2*nline)
      diffcux = CuArray{eltype(cfg),1,Nothing}(x)
      Pg = diffcux[1:nPg]
      Qg = diffcux[nPg+1:nPg+nQg]
      Va = diffcux[nPg+nQg+1:nPg+nQg+nVa]
      Vm = diffcux[nPg+nQg+nVa+1:end]
      arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, timeroutput)
      fjac = cux -> ForwardDiff.jacobian(constraints, cux, cfg)
      jac = fjac(cux)
      k = 1
      for i in 1:m
        for j in 1:n
          values[k] = jac[i,j]
          k += 1
        end
      end
      myprint("jac_x", x)
      myprint("jac_g", values)
    end
    end
    return
  end

  function eval_h(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Vector{Float64})
    @timeit timeroutput "eval_h" begin
    if mode == :Structure
      idx = 1
      for row = 1:n
        for col = 1:row
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
        arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, timeroutput)
        return acopf.objective(opfdata, arrays, timeroutput)
      end
      
      @timeit timeroutput "moving to GPU" begin
      cux = CuArray{Float64,1,Nothing}(x)
      end
      h = cux -> ForwardDiff.hessian(objective, cux)
      objhess = h(cux)
      @timeit timeroutput "moving from GPU objective" begin
      k = 1
      for i in 1:n
        for j in 1:i
          values[k] = obj_factor * objhess[i,j] 
          k += 1
        end
      end
      end
      function constraints(x)
        Pg = x[1:nPg]
        Qg = x[nPg+1:nPg+nQg]
        Va = x[nPg+nQg+1:nPg+nQg+nVa]
        Vm = x[nPg+nQg+nVa+1:end]
        # arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata)
        acopf.update_arrays!(arrays, Pg, Qg, Va, Vm, opfdata, timeroutput)
        T = typeof(x)
        # println("T inside function: ", T)
        # rbalconst   = T(undef, nbus)
        # ibalconst   = T(undef, nbus)
        # limitsto    = T(undef, nline)
        # limitsfrom  = T(undef, nline)
        acopf.constraints(rbalconst, ibalconst, limitsto, limitsfrom, opfdata, arrays, timeroutput)
        # y = T(undef, 2*nbus+2*nline)
        y[1:nbus] = rbalconst[:] 
        y[nbus+1:2*nbus] = ibalconst[:] 
        y[2*nbus+1:2*nbus+nline] = limitsto[:] 
        y[2*nbus+nline+1:end] = limitsfrom[:] 
        return y
      end
      @timeit timeroutput "remaining" begin
      cfg1 = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk(x))
      cfg = ForwardDiff.JacobianConfig(nothing, cfg1.duals, ForwardDiff.Chunk(x))
      rbalconst   = CuArray{eltype(cfg), 1, Nothing}(undef, nbus)
      ibalconst   = CuArray{eltype(cfg), 1, Nothing}(undef, nbus)
      limitsto    = CuArray{eltype(cfg), 1, Nothing}(undef, nline)
      limitsfrom  = CuArray{eltype(cfg), 1, Nothing}(undef, nline)
      y = CuArray{eltype(cfg)}(undef, 2*nbus+2*nline)
      diffcux = CuArray{eltype(cfg),1,Nothing}(x)
      Pg = diffcux[1:nPg]
      Qg = diffcux[nPg+1:nPg+nQg]
      Va = diffcux[nPg+nQg+1:nPg+nQg+nVa]
      Vm = diffcux[nPg+nQg+nVa+1:end]
      end
      @timeit timeroutput "compute hessian" begin
      arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, timeroutput)
      fhess = x -> ForwardDiff.jacobian(x -> ForwardDiff.jacobian(constraints, x), x)
      hess = fhess(cux)
      end
      @timeit timeroutput "reshape" begin
      hess = reshape(hess, m, n, n)
      end
      
     
      @timeit timeroutput "moving from GPU constraint" begin
      for l in 1:m
        k = 1
        for i in 1:n
          for j in 1:i
            values[k] += lambda[l] * hess[l,i,j]
            k += 1
          end
        end
      end
      end
      myprint("h_x", x)
      myprint("h", values)
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
    x_L[nPg + nQg + i] = -Inf
    x_U[nPg + nQg + i] = Inf
    x_L[nPg + nQg + nVa + i] = b.Vmin
    x_U[nPg + nQg + nVa + i] = b.Vmax
  end
  x_L[nPg+nQg+1] = 0.0
  x_U[nPg+nQg+1] = 0.0

  g_L = Vector{Float64}(undef, m)
  g_U = Vector{Float64}(undef, m)
  for i in 1:2*nbus g_L[i] = 0.0 end
  for i in 1:2*nbus g_U[i] = 0.0 end
  for i in 2*nbus+1:m g_L[i] = -Inf end
  for i in 2*nbus+1:m g_U[i] = 0.0 end
  
  myprint("x_L", x_L)
  myprint("x_U", x_U)

  myprint("g_L", g_L)
  myprint("g_U", g_U)

  idx = 1
  for row = 1:n
    for col = 1:row
      idx += 1
    end
  end
  prob = createProblem(n, x_L, x_U, m, g_L, g_U, m*n, idx-1,
                      eval_f, eval_g,eval_grad_f, eval_jac_g, eval_h)
  prob.x = 	[0.0 for i in 1:n]
  # # This tests callbacks.
  function intermediate(alg_mod::Int, iter_count::Int,
  obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
  d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
  ls_trials::Int)
  return iter_count < max_iter  # Interrupts after one iteration.
  end

  setIntermediateCallback(prob, intermediate)

  solvestat = solveProblem(prob)
  close(io)
  # @test Ipopt.ApplicationReturnStatus[solvestat] == :User_Requested_Stop
  return

end
end