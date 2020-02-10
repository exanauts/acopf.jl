using Revise
include("acopf.jl")
include("IpoptTest.jl")
using TimerOutputs
using CuArrays, CUDAnative
using JuMP
timeroutput = TimerOutput()
using ForwardDiff
import .acopf
import .IpoptTest
using Profile

# case="acopf/data/case9241pegase"
# case="acopf/data/case1354pegase"
# case="acopf/data/case9"
case="acopf/data/case30"
# case="acopf/data/case118"

function main()
max_iter=1
println("IPOPT JUMP")
@timeit timeroutput "load" begin
  opfdata = acopf.opf_loaddata(case)
end
@timeit timeroutput "model" begin
  opfmodel, Pg, Qg, Va, Vm = acopf.model(opfdata; max_iter = max_iter)
end
# @timeit timeroutput "model ad" begin
#   fobjective, fbalance = acopf_model_ad(opfdata)
# end
@timeit timeroutput "solve" begin
  opfmodel,status = acopf.solve(opfmodel,opfdata)
end

if status==MOI.LOCALLY_SOLVED
  acopf.outputAll(opfmodel,opfdata, Pg, Qg, Va, Vm)
end
@show size(Pg,1)
# Pg0 = value.(Pg) ; Qg0 = value.(Qg) ; Vm0 = value.(Vm) ; Va0 = value.(Va)
Pg0, Qg0, Vm0, Va0 = acopf.initialPt_IPOPT(opfdata)
# t1sPg, t2sPg = acopf.benchmark(Pg0, Qg0, Vm0, Va0, 3, 3, 0, timeroutput, opfdata)
println("IPOPT TEST")
# Profile.init(;n=100000000)
# @profile IpoptTest.test(Pg0, Qg0, Vm0, Va0, 3, 3, timeroutput, case; max_iter = max_iter)
# @profile IpoptTest.test(Pg0, Qg0, Vm0, Va0, 3, 3, timeroutput, case; max_iter = max_iter)

@timeit timeroutput "warmup" begin
IpoptTest.test(Pg0, Qg0, Vm0, Va0, 3, 3, timeroutput, case; max_iter = max_iter)
end
@timeit timeroutput "benchmark" begin
IpoptTest.test(Pg0, Qg0, Vm0, Va0, 3, 3, timeroutput, case; max_iter = max_iter)
end
# t1sPg, t1sPg = acopf.benchmark(opfdata, Pg, Qg, Vm, Va, size(Pg,1), size(Pg,1), 100, timeroutput)
# t1sPg, t1sPg = acopf.benchmark(opfdata, Pg, Qg, Vm, Va, 10, 10, 100, timeroutput)
# println("Objective: ", ForwardDiff.value.(t1sPg))
# println("Objective gradient: ", ForwardDiff.partials.(t1sPg).values)
# println("Objective Hessian: ", [i.values for i in ForwardDiff.partials.(ForwardDiff.partials.(t2sPg).values)])
# println("Constraint Hessian: ", ForwardDiff.partials.(t2srbalconst))
show(timeroutput)
# io = open("profile.out", "w+")
# Profile.print(io;maxdepth=4)
# close(io)
return
end

main()
