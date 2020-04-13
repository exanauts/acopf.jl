module IpoptTest
using Ipopt
using acopf
using ForwardDiff
using CuArrays, CUDAnative
using TimerOutputs
using SparseDiffTools
using SparseArrays

function test(Pg0, Qg0, Vm0, Va0, timeroutput, opfdata, arraytype; max_iter=100)

  if arraytype == CuArray
    T = CuVector
  elseif arraytype == Array
    T = Vector
  else
    error("Unkown array type $arraytype.")
  end

  nPg = size(Pg0,1) ; nQg = size(Qg0,1) ; nVm = size(Vm0,1) ; nVa = size(Va0,1)
  n = nPg + nQg + nVm + nVa
  nbus = length(opfdata.buses)
  nline = length(opfdata.lines)
  ngen = length(opfdata.generators)
  m = 2 * nbus + 2 * nline

  function get_jac_sparsity()
    println("Computing coloring...")
    x = Vector{Float64}(ones(Float64, n))  
    Pg = Vector{Float64}(zeros(Float64, nPg))
    Qg = Vector{Float64}(zeros(Float64, nQg))
    Va = Vector{Float64}(zeros(Float64, nVa))
    Vm = Vector{Float64}(zeros(Float64, nVm))
    rbalconst = Vector{Float64}(zeros(Float64, nbus))
    ibalconst = Vector{Float64}(zeros(Float64, nbus))
    limitsto = Vector{Float64}(zeros(Float64, nline))
    limitsfrom = Vector{Float64}(zeros(Float64, nline))
    function constraints(x)
      Pg = x[1:nPg]
      Qg = x[nPg+1:nPg+nQg]
      Va = x[nPg+nQg+1:nPg+nQg+nVa]
      Vm = x[nPg+nQg+nVa+1:end]
      acopf.update_arrays!(arrays, Pg, Qg, Va, Vm, timeroutput)
      acopf.constraints(rbalconst, ibalconst, limitsto, limitsfrom, arrays, timeroutput)
      y[1:nbus] = rbalconst[:] 
      y[nbus+1:2*nbus] = ibalconst[:] 
      y[2*nbus+1:2*nbus+nline] = limitsto[:] 
      y[2*nbus+nline+1:end] = limitsfrom[:] 
      return y
    end
    cux = Vector{Float64}(x)
    cfg = ForwardDiff.JacobianConfig(constraints, cux)
    rbalconst   = Vector{eltype(cfg)}(undef, nbus)
    ibalconst   = Vector{eltype(cfg)}(undef, nbus)
    limitsto    = Vector{eltype(cfg)}(undef, nline)
    limitsfrom  = Vector{eltype(cfg)}(undef, nline)
    y = Vector{eltype(cfg)}(undef, 2*nbus+2*nline)
    diffcux = Vector{eltype(cfg)}(x)
    Pg = diffcux[1:nPg]
    Qg = diffcux[nPg+1:nPg+nQg]
    Va = diffcux[nPg+nQg+1:nPg+nQg+nVa]
    Vm = diffcux[nPg+nQg+nVa+1:end]
    arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, timeroutput, Array)
    fjac = cux -> ForwardDiff.jacobian(constraints, cux, cfg)
    jac = sparse(fjac(cux))
    colors = unique(matrix_colors(jac))
    println("Done.")
    return colors
  end
  cuPg = T{Float64}(zeros(Float64, nPg))
  cuQg = T{Float64}(zeros(Float64, nQg))
  cuVa = T{Float64}(zeros(Float64, nVa))
  cuVm = T{Float64}(zeros(Float64, nVm))
  curbalconst = T{Float64}(zeros(Float64, nbus))
  cuibalconst = T{Float64}(zeros(Float64, nbus))
  culimitsto = T{Float64}(zeros(Float64, nline))
  culimitsfrom = T{Float64}(zeros(Float64, nline))
  # println("Number of Jacobian colors: ", size(get_jac_sparsity(),1))
  # return
  io = open("log.out", "w+")
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
    arrays = acopf.create_arrays(cuPg, cuQg, cuVa, cuVm, opfdata, timeroutput, arraytype)
    obj = acopf.objective(arrays, timeroutput)
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
    arrays = acopf.create_arrays(cuPg, cuQg, cuVa, cuVm, opfdata, timeroutput, arraytype)
    acopf.constraints(curbalconst, cuibalconst, culimitsto, culimitsfrom, arrays, timeroutput)
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
      arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, timeroutput, arraytype)
      return acopf.objective(arrays, timeroutput)
    end
    cux = T{Float64}(x)
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
        acopf.update_arrays!(arrays, Pg, Qg, Va, Vm, timeroutput)
        acopf.constraints(rbalconst, ibalconst, limitsto, limitsfrom, arrays, timeroutput)
        y[1:nbus] = rbalconst[:] 
        y[nbus+1:2*nbus] = ibalconst[:] 
        y[2*nbus+1:2*nbus+nline] = limitsto[:] 
        y[2*nbus+nline+1:end] = limitsfrom[:] 
        return y
      end
      cux = T{Float64}(x)
      cfg = ForwardDiff.JacobianConfig(constraints, cux)
      rbalconst   = T{eltype(cfg)}(undef, nbus)
      ibalconst   = T{eltype(cfg)}(undef, nbus)
      limitsto    = T{eltype(cfg)}(undef, nline)
      limitsfrom  = T{eltype(cfg)}(undef, nline)
      y = T{eltype(cfg)}(undef, 2*nbus+2*nline)
      diffcux = T{eltype(cfg)}(x)
      Pg = diffcux[1:nPg]
      Qg = diffcux[nPg+1:nPg+nQg]
      Va = diffcux[nPg+nQg+1:nPg+nQg+nVa]
      Vm = diffcux[nPg+nQg+nVa+1:end]
      arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, timeroutput, arraytype)
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
        arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, timeroutput, arraytype)
        return acopf.objective(arrays, timeroutput)
      end
      
      @timeit timeroutput "moving to GPU" begin
      cux = T{Float64}(x)
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
        acopf.update_arrays!(arrays, Pg, Qg, Va, Vm, timeroutput)
        acopf.constraints(rbalconst, ibalconst, limitsto, limitsfrom, arrays, timeroutput)
        y[1:nbus] = rbalconst[:] 
        y[nbus+1:2*nbus] = ibalconst[:] 
        y[2*nbus+1:2*nbus+nline] = limitsto[:] 
        y[2*nbus+nline+1:end] = limitsfrom[:] 
        return y
      end
      @timeit timeroutput "remaining" begin
      cfg1 = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk(x))
      cfg = ForwardDiff.JacobianConfig(nothing, cfg1.duals, ForwardDiff.Chunk(x))
      rbalconst   = T{eltype(cfg)}(undef, nbus)
      ibalconst   = T{eltype(cfg)}(undef, nbus)
      limitsto    = T{eltype(cfg)}(undef, nline)
      limitsfrom  = T{eltype(cfg)}(undef, nline)
      y = T{eltype(cfg)}(undef, 2*nbus+2*nline)
      diffcux = T{eltype(cfg)}(x)
      Pg = diffcux[1:nPg]
      Qg = diffcux[nPg+1:nPg+nQg]
      Va = diffcux[nPg+nQg+1:nPg+nQg+nVa]
      Vm = diffcux[nPg+nQg+nVa+1:end]
      end
      @timeit timeroutput "compute hessian" begin
      arrays = acopf.create_arrays(Pg, Qg, Va, Vm, opfdata, timeroutput, arraytype)
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

  # Callback
  function intermediate(alg_mod::Int, iter_count::Int,
  obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
  d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
  ls_trials::Int)
  return iter_count < max_iter  # Interrupts after one iteration.
  end

  setIntermediateCallback(prob, intermediate)

  solvestat = solveProblem(prob)
  close(io)
  return

end
end