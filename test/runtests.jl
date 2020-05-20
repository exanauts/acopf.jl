using Test
using acopf
using TimerOutputs

@testset "Ipopt CPU evaluation" begin
  arraytype = Array
  include("Ipopt.jl")
  case="data/case9"
  max_iter=100
  opfdata = acopf.opf_loaddata(case)
  Pg0, Qg0, Vm0, Va0 = acopf.initialPt_IPOPT(opfdata)
  obj_val = test(Pg0, Qg0, Vm0, Va0, timeroutput, opfdata, arraytype; max_iter = max_iter)
  @test obj_val ≈ 5.2966862e+03
end

@testset "Ipopt GPU evaluation" begin
  arraytype = CuArray
  include("Ipopt.jl")
  case="data/case9"
  max_iter=100
  opfdata = acopf.opf_loaddata(case)
  Pg0, Qg0, Vm0, Va0 = acopf.initialPt_IPOPT(opfdata)
  obj_val = test(Pg0, Qg0, Vm0, Va0, timeroutput, opfdata, arraytype; max_iter = max_iter)
  @test obj_val ≈ 5.2966862e+03
end

@testset "JuMP ACOPF Reference" begin
  include("jumpmodel.jl")
  case="data/case9"
  max_iter=100
  opfdata = acopf.opf_loaddata(case)
  Pg0, Qg0, Vm0, Va0 = acopf.initialPt_IPOPT(opfdata)
  opfmodel, Pg, Qg, Va, Vm = model(opfdata; max_iter = max_iter)
  opfmodel,status = solve(opfmodel,opfdata)
  if status==MOI.LOCALLY_SOLVED
    outputAll(opfmodel,opfdata, Pg, Qg, Va, Vm)
  end
  @test objective_value(opfmodel) ≈ 5.2966862e+03
end