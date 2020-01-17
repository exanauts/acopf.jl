include("acopf.jl")
using TimerOutputs
using CuArrays, CUDAnative
timeroutput = TimerOutput()

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
cuPg = CuArray{Float64,1,Nothing}(value.(Pg))
cuQg = CuArray{Float64,1,Nothing}(value.(Qg))
cuVa = CuArray{Float64,1,Nothing}(value.(Va))
cuVm = CuArray{Float64,1,Nothing}(value.(Vm))
# Pg0,Qg0,Vm0,Va0 = acopf_initialPt_IPOPT(opfdata)
# cuPg = CuArray{Float64,1,Nothing}(Pg0)
# cuQg = CuArray{Float64,1,Nothing}(Qg0)
# cuVa = CuArray{Float64,1,Nothing}(Va0)
# cuVm = CuArray{Float64,1,Nothing}(Vm0)
rbalconst = CuArray{Float64,1,Nothing}(undef, size(Va,1))
ibalconst = CuArray{Float64,1,Nothing}(undef, size(Va,1))
@show vPg = fobjective(cuPg, opfdata)
balance(rbalconst, ibalconst, cuPg, cuQg, cuVa, cuVm, opfdata)
@show rbalconst
@show ibalconst
  if status==MOI.LOCALLY_SOLVED
    acopf_outputAll(opfmodel,opfdata, Pg, Qg, Vm, Va)
  end
show(timeroutput)
end

main()
