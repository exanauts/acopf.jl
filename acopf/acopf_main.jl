using Revise
include("acopf.jl")
include("IpoptTest.jl")
include("jumpmodel.jl")
using TimerOutputs
timeroutput = TimerOutput()
using CuArrays, CUDAnative
using JuMP
using ForwardDiff
import .acopf
import .IpoptTest
import .jumpmodel

# case="acopf/data/case9241pegase"
# case="acopf/data/case1354pegase"
case="acopf/data/case9"
# case="acopf/data/case30"
# case="acopf/data/case118"
function jump_model(opfdata, max_iter)
  println("Ipopt through JuMP")
  @timeit timeroutput "model" begin
    opfmodel, Pg, Qg, Va, Vm = jumpmodel.model(opfdata; max_iter = max_iter)
  end
  @timeit timeroutput "solve" begin
    opfmodel,status = jumpmodel.solve(opfmodel,opfdata)
  end

  if status==MOI.LOCALLY_SOLVED
    jumpmodel.outputAll(opfmodel,opfdata, Pg, Qg, Va, Vm)
  end
  return Pg, Qg, Vm, Va
end
# JuMP model
function main()
  max_iter=100
  arraytype = CuArray
  @timeit timeroutput "load data" begin
  opfdata = jumpmodel.opf_loaddata(case)
  end
  Pg, Qg, Vm, Va = jump_model(opfdata, max_iter)
  # Pg0 = value.(Pg) ; Qg0 = value.(Qg) ; Vm0 = value.(Vm) ; Va0 = value.(Va)
  Pg0, Qg0, Vm0, Va0 = jumpmodel.initialPt_IPOPT(opfdata)
  # t1sPg, t2sPg = acopf.benchmark(Pg0, Qg0, Vm0, Va0, 3, 3, 0, timeroutput, opfdata)
  println("Ipopt interface")
  @timeit timeroutput "warmup" begin
  IpoptTest.test(Pg0, Qg0, Vm0, Va0, timeroutput, opfdata, arraytype; max_iter = max_iter)
  end
  # reset_timer!(timeroutput)
  # reset_timer!()
  @timeit timeroutput "benchmark" begin
  IpoptTest.test(Pg0, Qg0, Vm0, Va0, timeroutput, opfdata, arraytype; max_iter = max_iter)
  end
  # t1sPg, t1sPg = acopf.benchmark(opfdata, Pg, Qg, Vm, Va, size(Pg,1), size(Pg,1), 100, timeroutput)
  acopf.benchmark(opfdata, Pg0, Qg0, Vm0, Va0, 3, 3, 100, timeroutput, arraytype)
  # println("Objective: ", ForwardDiff.value.(t1sPg))
  # println("Objective gradient: ", ForwardDiff.partials.(t1sPg).values)
  # println("Objective Hessian: ", [i.values for i in ForwardDiff.partials.(ForwardDiff.partials.(t2sPg).values)])
  # println("Constraint Hessian: ", ForwardDiff.partials.(t2srbalconst))
  show(timeroutput)
  print_timer()
  return
end

main()
