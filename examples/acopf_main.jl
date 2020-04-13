using TimerOutputs
timeroutput = TimerOutput()
using CuArrays, CUDAnative
using JuMP
using ForwardDiff
using acopf

# case="data/case9241pegase"
# case="data/case1354pegase"
case="data/case9"
# case="data/case30"
# case="data/case118"
function jump_model(opfdata, max_iter)
  println("Ipopt through JuMP")
  @timeit timeroutput "model" begin
    opfmodel, Pg, Qg, Va, Vm = acopf.jumpmodel.model(opfdata; max_iter = max_iter)
  end
  @timeit timeroutput "solve" begin
    opfmodel,status = acopf.jumpmodel.solve(opfmodel,opfdata)
  end

  if status==MOI.LOCALLY_SOLVED
    acopf.jumpmodel.outputAll(opfmodel,opfdata, Pg, Qg, Va, Vm)
  end
  return Pg, Qg, Vm, Va
end
function main()
  max_iter=100
  arraytype = Array
  @timeit timeroutput "load data" begin
  opfdata = acopf.jumpmodel.opf_loaddata(case)
  end
  Pg, Qg, Vm, Va = jump_model(opfdata, max_iter)
  Pg0, Qg0, Vm0, Va0 = acopf.jumpmodel.initialPt_IPOPT(opfdata)
  println("Ipopt interface")
  @timeit timeroutput "warmup" begin
  acopf.IpoptTest.test(Pg0, Qg0, Vm0, Va0, timeroutput, opfdata, arraytype; max_iter = max_iter)
  end
  # reset_timer!(timeroutput)
  # reset_timer!()
  @timeit timeroutput "benchmark" begin
  acopf.IpoptTest.test(Pg0, Qg0, Vm0, Va0, timeroutput, opfdata, arraytype; max_iter = max_iter)
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
