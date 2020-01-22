using Revise
include("acopf.jl")
using TimerOutputs
using CuArrays, CUDAnative
using JuMP
timeroutput = TimerOutput()
using ForwardDiff
import .acopf

case="acopf/data/case9241pegase"
# case="acopf/data/case9"
# case="acopf/data/case118"

function main()

@timeit timeroutput "load" begin
  opfdata = acopf.opf_loaddata(case)
end
@timeit timeroutput "model" begin
  opfmodel, Pg, Qg, Vm, Va = acopf.acopf_model(opfdata)
end
# @timeit timeroutput "model ad" begin
#   fobjective, fbalance = acopf_model_ad(opfdata)
# end
@timeit timeroutput "solve" begin
  opfmodel,status = acopf.acopf_solve(opfmodel,opfdata)
end

if status==MOI.LOCALLY_SOLVED
  acopf.acopf_outputAll(opfmodel,opfdata, Pg, Qg, Vm, Va)
end
@show size(Pg,1)
t1sPg = acopf.benchmark(opfdata, Pg, Qg, Vm, Va, 1, 1, 10, timeroutput)
# println("Objective: ", ForwardDiff.value.(t1sPg))
println("Objective gradient: ", ForwardDiff.partials.(t1sPg))
# println("Objective Hessian: ", ForwardDiff.partials.(t2sPg))
# println("Constraint Hessian: ", ForwardDiff.partials.(t2srbalconst))
show(timeroutput)
return opfmodel
end

opfmodel = main()
