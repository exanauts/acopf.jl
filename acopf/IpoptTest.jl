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
  cuVm = CuArray{Float64,1,Nothing}(zeros(Float64, nVm))
  cuVa = CuArray{Float64,1,Nothing}(zeros(Float64, nVa))
  curbalconst = CuArray{Float64,1,Nothing}(zeros(Float64, nbus))
  cuibalconst = CuArray{Float64,1,Nothing}(zeros(Float64, nbus))
  culimitsto = CuArray{Float64,1,Nothing}(zeros(Float64, nline))
  culimitsfrom = CuArray{Float64,1,Nothing}(zeros(Float64, nline))
  function eval_f(x::Vector{Float64})
    cuPg[:] = x[1:nPg] 
    cuQg[:] = x[nPg+1:nPg+nQg] 
    cuVm[:] = x[nPg+nQg+1:nPg+nQg+nVm] 
    cuVa[:] = x[nPg+nQg+nVm+1:end] 
    arrays = acopf.create_arrays(cuPg, cuQg, cuVa, cuVm, opfdata, Float64)
    return acopf.objective(opfdata, arrays)
    # return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
  end

  function eval_g(x::Vector{Float64}, g::Vector{Float64})
    cuPg[:] = x[1:nPg] 
    cuQg[:] = x[nPg+1:nPg+nQg] 
    cuVm[:] = x[nPg+nQg+1:nPg+nQg+nVm] 
    cuVa[:] = x[nPg+nQg+nVm+1:end] 
    arrays = acopf.create_arrays(cuPg, cuQg, cuVa, cuVm, opfdata, Float64)
    @show x
    @show cuVm
    acopf.constraints(curbalconst, cuibalconst, culimitsto, culimitsfrom, opfdata, arrays, timeroutput)
    g[1:nbus] = curbalconst[:]
    g[nbus+1:2*nbus] = cuibalconst[:]
    g[2*nbus+1:2*nbus+nline] = culimitsto[:]
    g[2*nbus+nline+1:end] = culimitsfrom[:]
    @show g
  end

  function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
    # Bad: grad_f    = zeros(4)  # Allocates new array
    # OK:  grad_f[:] = zeros(4)  # Modifies 'in place'
    grad_f[1] = x[1] * x[4] + x[4] * (x[1] + x[2] + x[3])
    grad_f[2] = x[1] * x[4]
    grad_f[3] = x[1] * x[4] + 1
    grad_f[4] = x[1] * (x[1] + x[2] + x[3])
  end

  function eval_jac_g(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
    if mode == :Structure
      # Constraint (row) 1
      rows[1] = 1; cols[1] = 1
      rows[2] = 1; cols[2] = 2
      rows[3] = 1; cols[3] = 3
      rows[4] = 1; cols[4] = 4
      # Constraint (row) 2
      rows[5] = 2; cols[5] = 1
      rows[6] = 2; cols[6] = 2
      rows[7] = 2; cols[7] = 3
      rows[8] = 2; cols[8] = 4
    else
      # Constraint (row) 1
      values[1] = x[2]*x[3]*x[4]  # 1,1
      values[2] = x[1]*x[3]*x[4]  # 1,2
      values[3] = x[1]*x[2]*x[4]  # 1,3
      values[4] = x[1]*x[2]*x[3]  # 1,4
      # Constraint (row) 2
      values[5] = 2*x[1]  # 2,1
      values[6] = 2*x[2]  # 2,2
      values[7] = 2*x[3]  # 2,3
      values[8] = 2*x[4]  # 2,4
    end
  end

  function eval_h(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Vector{Float64})
    if mode == :Structure
      # Symmetric matrix, fill the lower left triangle only
      idx = 1
      for row = 1:4
        for col = 1:row
          rows[idx] = row
          cols[idx] = col
          idx += 1
        end
      end
    else
      # Again, only lower left triangle
      # Objective
      values[1] = obj_factor * (2*x[4])  # 1,1
      values[2] = obj_factor * (  x[4])  # 2,1
      values[3] = 0                      # 2,2
      values[4] = obj_factor * (  x[4])  # 3,1
      values[5] = 0                      # 3,2
      values[6] = 0                      # 3,3
      values[7] = obj_factor * (2*x[1] + x[2] + x[3])  # 4,1
      values[8] = obj_factor * (  x[1])  # 4,2
      values[9] = obj_factor * (  x[1])  # 4,3
      values[10] = 0                     # 4,4

      # First constraint
      values[2] += lambda[1] * (x[3] * x[4])  # 2,1
      values[4] += lambda[1] * (x[2] * x[4])  # 3,1
      values[5] += lambda[1] * (x[1] * x[4])  # 3,2
      values[7] += lambda[1] * (x[2] * x[3])  # 4,1
      values[8] += lambda[1] * (x[1] * x[3])  # 4,2
      values[9] += lambda[1] * (x[1] * x[2])  # 4,3

      # Second constraint
      values[1]  += lambda[2] * 2  # 1,1
      values[3]  += lambda[2] * 2  # 2,2
      values[6]  += lambda[2] * 2  # 3,3
      values[10] += lambda[2] * 2  # 4,4
    end
  end
  # n = 4
  # x_L = [1.0, 1.0, 1.0, 1.0]
  # x_U = [5.0, 5.0, 5.0, 5.0]
  x_L = [-2.0e19 for i in 1:n]
  x_U = [ 2.0e19 for i in 1:n]

  g_L = Vector{Float64}(undef, m)
  g_U = Vector{Float64}(undef, m)
  for i in 1:2*nbus g_L[i] = 0.0 end
  for i in 1:2*nbus g_U[i] = 0.0 end
  for i in 2*nbus+1:m g_L[i] = -2.0e19 end
  for i in 2*nbus+1:m g_U[i] = 0.0 end

  prob = createProblem(n, x_L, x_U, m, g_L, g_U, 8, 10,
                      eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
  @show Pg0
  @show Qg0
  @show Vm0
  @show Va0
  prob.x[1:nPg] = Pg0[:] 
  prob.x[nPg+1:nPg+nQg] = Qg0[:]
  prob.x[nPg+nQg+1:nPg+nQg+nVm] = Vm0[:] 
  prob.x[nPg+nQg+nVm+1:end] = Va0[:]

  @show prob.x
  @show eval_f(prob.x)
  # g = Vector{Float64}(undef, 2*nbus+2*nline)
  g = zeros(Float64, 2*nbus+2*nline)
  eval_g(prob.x, g)
  @show g
  
  # # This tests callbacks.
  # function intermediate(alg_mod::Int, iter_count::Int,
  # obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
  # d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
  # ls_trials::Int)
  # return iter_count < 1  # Interrupts after one iteration.
  # end

  # setIntermediateCallback(prob, intermediate)

  # solvestat = solveProblem(prob)

  # @test Ipopt.ApplicationReturnStatus[solvestat] == :User_Requested_Stop

end
end