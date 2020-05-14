using Test
using acopf
using TimerOutputs

@testset "Ipopt CPU evaluation" begin
  arraytype = Array
  include("Ipopt.jl")
  case="data/case9"
  max_iter=100
  opfdata = acopf.jumpmodel.opf_loaddata(case)
  Pg0, Qg0, Vm0, Va0 = acopf.jumpmodel.initialPt_IPOPT(opfdata)
  obj_val = test(Pg0, Qg0, Vm0, Va0, timeroutput, opfdata, arraytype; max_iter = max_iter)
  @test obj_val ≈ 5.2966862e+03
end

@testset "Ipopt GPU evaluation" begin
  arraytype = CuArray
  include("Ipopt.jl")
  case="data/case9"
  max_iter=100
  opfdata = acopf.jumpmodel.opf_loaddata(case)
  Pg0, Qg0, Vm0, Va0 = acopf.jumpmodel.initialPt_IPOPT(opfdata)
  obj_val = test(Pg0, Qg0, Vm0, Va0, timeroutput, opfdata, arraytype; max_iter = max_iter)
  @test obj_val ≈ 5.2966862e+03
end