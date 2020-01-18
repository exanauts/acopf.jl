include("acopf.jl")
using TimerOutputs
using CuArrays, CUDAnative
timeroutput = TimerOutput()
using ForwardDiff

case="acopf/case9"

function main()

@timeit timeroutput "load" begin
  opfdata = opf_loaddata(case)
end
@timeit timeroutput "model" begin
  opfmodel, Pg, Qg, Vm, Va = acopf_model(opfdata)
end
# @timeit timeroutput "model ad" begin
#   fobjective, fbalance = acopf_model_ad(opfdata)
# end
@timeit timeroutput "solve" begin
  opfmodel,status = acopf_solve(opfmodel,opfdata)
end
cuPg = CuArray(ForwardDiff.Dual.(value.(Pg), 1))
cuQg = CuArray(ForwardDiff.Dual.(value.(Qg), 1))
cuVa = CuArray(ForwardDiff.Dual.(value.(Va), 1))
cuVm = CuArray(ForwardDiff.Dual.(value.(Vm), 1))
# Pg0,Qg0,Vm0,Va0 = acopf_initialPt_IPOPT(opfdata)
# cuPg = CuArray{Float64,1,Nothing}(Pg0)
# cuQg = CuArray{Float64,1,Nothing}(Qg0)
# cuVa = CuArray{Float64,1,Nothing}(Va0)
# cuVm = CuArray{Float64,1,Nothing}(Vm0)
T = typeof(cuPg)
rbalconst = T(undef, size(Va,1))
ibalconst = T(undef, size(Va,1))
limitsto = T(undef, size(Va,1))
limitsfrom = T(undef, size(Va,1))

@show vPg = objective(cuPg, opfdata)
constraints(rbalconst, ibalconst, limitsto, limitsfrom, cuPg, cuQg, cuVa, cuVm, opfdata)
@show rbalconst
@show ibalconst
@show limitsto
@show limitsfrom
  if status==MOI.LOCALLY_SOLVED
    acopf_outputAll(opfmodel,opfdata, Pg, Qg, Vm, Va)
  end
show(timeroutput)
end

main()
