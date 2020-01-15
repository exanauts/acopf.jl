include("acopf.jl")
using TimerOutputs
timeroutput = TimerOutput()

case="acopf/case9"

function main()

@timeit timeroutput "load" begin
  opfdata = opf_loaddata(case)
end
@timeit timeroutput "model" begin
  opfmodel, Pg, Qg, Vm, Va = acopf_model(opfdata)
end
@timeit timeroutput "solve" begin
  opfmodel,status = acopf_solve(opfmodel,opfdata)
end
  if status==MOI.LOCALLY_SOLVED
    acopf_outputAll(opfmodel,opfdata, Pg, Qg, Vm, Va)
  end
show(timeroutput)
end

main()
