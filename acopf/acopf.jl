module acopf

include("opfdata.jl")
using JuMP
using Ipopt
using Printf
using DelimitedFiles
using CuArrays, CUDAnative
using ForwardDiff
using TimerOutputs

export acopf_solve, opf_loaddata, acopf_model
export create_arrays, objective, acopf_outputAll, constraints 
export acopf_initialPt_IPOPT
export myseed!

function acopf_solve(opfmodel, opf_data)
   
  # 
  # Initial point - needed especially for pegase cases
  #
  Pg0,Qg0,Vm0,Va0 = acopf_initialPt_IPOPT(opf_data)
  opfmodel[:Pg] = Pg0  
  opfmodel[:Qg] = Qg0
  opfmodel[:Vm] = Vm0
  opfmodel[:Va] = Va0

  optimize!(opfmodel)
  status = termination_status(opfmodel)
  if status != MOI.LOCALLY_SOLVED
    println("Could not solve the model to optimality.")
  end
  return opfmodel,status
end

function acopf_model(opf_data)
  #shortcuts for compactness
  lines = opf_data.lines; buses = opf_data.buses; generators = opf_data.generators; baseMVA = opf_data.baseMVA
  busIdx = opf_data.BusIdx; FromLines = opf_data.FromLines; ToLines = opf_data.ToLines; BusGeners = opf_data.BusGenerators;

  nbus  = length(buses); nline = length(lines); ngen  = length(generators)

  #branch admitances
  YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)

  #
  # JuMP model now
  #
  opfmodel = Model(with_optimizer(Ipopt.Optimizer))

  @variable(opfmodel, generators[i].Pmin <= Pg[i=1:ngen] <= generators[i].Pmax)
  @variable(opfmodel, generators[i].Qmin <= Qg[i=1:ngen] <= generators[i].Qmax)

  @variable(opfmodel, buses[i].Vmin <= Vm[i=1:nbus] <= buses[i].Vmax)
  @variable(opfmodel, Va[1:nbus])
  #fix the voltage angle at the reference bus
  set_lower_bound(Va[opf_data.bus_ref], buses[opf_data.bus_ref].Va)
  set_upper_bound(Va[opf_data.bus_ref], buses[opf_data.bus_ref].Va)

  # minimize active power
  coeff0 = Vector{Float64}(undef, ngen)
  for (i,v) in enumerate(generators)
    coeff0[i] = v.coeff[v.n]
  end
  coeff1 = Vector{Float64}(undef, ngen)
  for (i,v) in enumerate(generators)
    coeff1[i] = v.coeff[v.n - 1]
  end
  coeff2 = Vector{Float64}(undef, ngen)
  for (i,v) in enumerate(generators)
    coeff2[i] = v.coeff[v.n - 2]
  end

  @NLobjective(opfmodel, Min, sum( coeff2[i]*(baseMVA*Pg[i])^2 
                                + coeff1[i]*(baseMVA*Pg[i])
                                + coeff0[i] for i=1:ngen))

  #
  # power flow balance
  #
  
  for b in 1:nbus
    #real part
    @NLconstraint(
      opfmodel, 
      ( sum( YffR[l] for l in FromLines[b]) + sum( YttR[l] for l in ToLines[b]) + YshR[b] ) * Vm[b]^2 
      + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *( YftR[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftI[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )  
      + sum( Vm[b]*Vm[busIdx[lines[l].from]]*( YtfR[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfI[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   ) 
      - ( sum(baseMVA*Pg[g] for g in BusGeners[b]) - buses[b].Pd ) / baseMVA      # Sbus part
      ==0)
    #imaginary part
    @NLconstraint(
      opfmodel,
      ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[b]^2 
      + sum( Vm[b]*Vm[busIdx[lines[l].to]]  *(-YftI[l]*cos(Va[b]-Va[busIdx[lines[l].to]]  ) + YftR[l]*sin(Va[b]-Va[busIdx[lines[l].to]]  )) for l in FromLines[b] )
      + sum( Vm[b]*Vm[busIdx[lines[l].from]]*(-YtfI[l]*cos(Va[b]-Va[busIdx[lines[l].from]]) + YtfR[l]*sin(Va[b]-Va[busIdx[lines[l].from]])) for l in ToLines[b]   )
      - ( sum(baseMVA*Qg[g] for g in BusGeners[b]) - buses[b].Qd ) / baseMVA      #Sbus part
      ==0)
  end
  #
  # branch/lines flow limits
  #
  nlinelim=0
  for l in 1:nline
    if lines[l].rateA!=0 && lines[l].rateA<1.0e10
      nlinelim += 1
      flowmax=(lines[l].rateA/baseMVA)^2

      #branch apparent power limits (from bus)
      Yff_abs2=YffR[l]^2+YffI[l]^2; Yft_abs2=YftR[l]^2+YftI[l]^2
      Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
      @NLconstraint(
        opfmodel,
	Vm[busIdx[lines[l].from]]^2 *
	( Yff_abs2*Vm[busIdx[lines[l].from]]^2 + Yft_abs2*Vm[busIdx[lines[l].to]]^2 
	  + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]*(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])-Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])) 
	) 
        - flowmax <=0)

      #branch apparent power limits (to bus)
      Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
      Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
      @NLconstraint(
        opfmodel,
	Vm[busIdx[lines[l].to]]^2 *
        ( Ytf_abs2*Vm[busIdx[lines[l].from]]^2 + Ytt_abs2*Vm[busIdx[lines[l].to]]^2
          + 2*Vm[busIdx[lines[l].from]]*Vm[busIdx[lines[l].to]]*(Yre*cos(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]])-Yim*sin(Va[busIdx[lines[l].from]]-Va[busIdx[lines[l].to]]))
        )
        - flowmax <=0)
    end
  end
  
  @printf("Buses: %d  Lines: %d  Generators: %d\n", nbus, nline, ngen)
  println("Lines with limits  ", nlinelim)
 
  return opfmodel, Pg, Qg, Va, Vm
end

struct CompArrays
  cuPg
  cuQg
  cuVa
  cuVm
  baseMVA
  nbus
  nline
  ngen
  coeff0
  coeff1
  coeff2
  viewToR
  viewFromR
  viewToI
  viewFromI
  viewVmTo
  viewVmFrom
  viewVaTo
  viewVaFrom
  viewYffR
  viewYttR
  viewYffI
  viewYttI
  cuYffR 
  cuYffI
  cuYttR
  cuYttI
  cuYftR
  cuYftI
  cuYtfR
  cuYtfI
  cuYshR
  cuYshI
  viewPg
  viewQg
  cuPd
  cuQd
  cuflowmax
  Yff_abs2 
  Yft_abs2
  Ytf_abs2
  Ytt_abs2
  Yrefrom
  Yimfrom
  Yreto
  Yimto
  viewVmToFromLines
  viewcuYftRFromLines
  viewVaToFromLines
  viewcuYftIFromLines
  viewVmFromToLines
  viewcuYtfRToLines
  viewVaFromToLines
  viewcuYtfIToLines
end

function create_arrays(cuPg::T, cuQg::T, cuVa::T, cuVm::T, opf_data::OPFData) where T
  #shortcuts for compactness
  lines = opf_data.lines; buses = opf_data.buses; generators = opf_data.generators; baseMVA = opf_data.baseMVA
  busIdx = opf_data.BusIdx; FromLines = opf_data.FromLines; ToLines = opf_data.ToLines; BusGeners = opf_data.BusGenerators;

  nbus  = length(buses); nline = length(lines); ngen  = length(generators)

  # Arrays for objective
  #branch admitances
  YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)
  coeff0 = CuArray{Float64,1,Nothing}(undef, ngen)
  for (i,v) in enumerate(generators)
    coeff0[i] = v.coeff[v.n]
  end
  coeff1 = CuArray{Float64,1,Nothing}(undef, ngen)
  for (i,v) in enumerate(generators)
    coeff1[i] = v.coeff[v.n - 1]
  end
  coeff2 = CuArray{Float64,1,Nothing}(undef, ngen)
  for (i,v) in enumerate(generators)
    coeff2[i] = v.coeff[v.n - 2]
  end
  
  # Arrays for constraints

  # demand arrays
  cuPd = CuArray{Float64,1,Nothing}(undef, nbus)
  for (b,v) in enumerate(cuPd)
    cuPd[b] = buses[b].Pd
  end
  cuQd = CuArray{Float64,1,Nothing}(undef, nbus)
  for (b,v) in enumerate(cuQd)
    cuQd[b] = buses[b].Qd
  end

  # Indices
  cuBusGeners = Array{CuArray{Int64,1,Nothing},1}(undef, nbus)
  for (b,v) in enumerate(BusGeners) cuBusGeners[b] = cu(BusGeners[b]) end
  cuFromLines = Array{CuArray{Int64,1,Nothing},1}(undef, nbus)
  for (b,v) in enumerate(FromLines) cuFromLines[b] = cu(FromLines[b]) end
  cuToLines = Array{CuArray{Int64,1,Nothing},1}(undef, nbus)
  for (b,v) in enumerate(ToLines) cuToLines[b] = cu(ToLines[b]) end

  # Views
  viewPg = T(undef, nbus)  
  for b in 1:nbus viewPg[b] = sum(view(cuPg, cuBusGeners[b])) end
  viewQg = T(undef, nbus)  
  for b in 1:nbus viewQg[b] = sum(view(cuQg, cuBusGeners[b])) end
  

  # Real power balance

  mapbus2lineto = CuArray{Int64,1,Nothing}([busIdx[lines[l].to] for l in 1:nline]) 
  mapbus2linefrom = CuArray{Int64,1,Nothing}([busIdx[lines[l].from] for l in 1:nline])
  viewVmTo = view(cuVm, mapbus2lineto)
  viewVmFrom = view(cuVm, mapbus2linefrom)
  viewVaTo = view(cuVa, mapbus2lineto)
  viewVaFrom = view(cuVa, mapbus2linefrom)

  # branch admitances
  cuYffR = CuArray{Float64,1,Nothing}(YffR) ; cuYffI = CuArray{Float64,1,Nothing}(YffI) ; cuYttR = CuArray{Float64,1,Nothing}(YttR) ; cuYttI = CuArray{Float64,1,Nothing}(YttI) ; cuYftR = CuArray{Float64,1,Nothing}(YftR)
  cuYftI = CuArray{Float64,1,Nothing}(YftI) ; cuYtfR = CuArray{Float64,1,Nothing}(YtfR) ; cuYtfI = CuArray{Float64,1,Nothing}(YtfI) ; cuYshR = CuArray{Float64,1,Nothing}(YshR) ; cuYshI = CuArray{Float64,1,Nothing}(YshI)

  # real views
  viewYffR = CuArray{Float64,1,Nothing}(undef, nbus) ; 
  for b in 1:nbus viewYffR[b] = sum(view(cuYffR, cuFromLines[b])) end
  viewYttR = CuArray{Float64,1,Nothing}(undef, nbus) ; 
  for b in 1:nbus viewYttR[b] = sum(view(cuYttR, cuToLines[b])) end
  # imaginary views
  viewYffI = CuArray{Float64,1,Nothing}(undef, nbus) ; 
  for b in 1:nbus viewYffI[b] = sum(.- view(cuYffI, cuFromLines[b])) end
  viewYttI = CuArray{Float64,1,Nothing}(undef, nbus) ; 
  for b in 1:nbus viewYttI[b] = sum(.- view(cuYttI, cuToLines[b])) end

  viewToR = T(undef, nbus) ; 
  viewFromR = T(undef, nbus) ; 
  viewToI = T(undef, nbus) ; 
  viewFromI = T(undef, nbus) ; 
  #
  # branch/lines flow limits
  #
  culinelimit = CuArray{Float64,1,Nothing}(undef, nline)
  nlinelimit = 0
  for l in 1:nline
    if lines[l].rateA!=0 && lines[l].rateA<1.0e10
      nlinelimit += 1
      culinelimit[l] = 1.0
    else
      culinelimit[l] = 0.0
    end
  end
  curate = CuArray{Float64,1,Nothing}(undef, nline)
  for l in 1:nline curate[l] = lines[l].rateA end
  cuflowmax= CuArray{Float64,1,Nothing}(undef, nline)
  cuflowmax .= (curate ./ baseMVA).^2 

  Yff_abs2=CuArray{Float64,1,Nothing}(undef, nline); Yft_abs2=CuArray{Float64,1,Nothing}(undef, nline); 
  Ytf_abs2=CuArray{Float64,1,Nothing}(undef, nline); Ytt_abs2=CuArray{Float64,1,Nothing}(undef, nline); 
  Yrefrom=CuArray{Float64,1,Nothing}(undef, nline); Yimfrom=CuArray{Float64,1,Nothing}(undef, nline); 
  Yreto=CuArray{Float64,1,Nothing}(undef, nline); Yimto=CuArray{Float64,1,Nothing}(undef, nline); 
  Yff_abs2 .= cuYffR.^2 .+ cuYffI.^2; Yft_abs2 .= cuYftR.^2 .+ cuYftI.^2
  Ytf_abs2 .= cuYtfR.^2 .+ cuYtfI.^2; Ytt_abs2 .= cuYttR.^2 .+ cuYttI.^2
  Yrefrom .= cuYffR .* cuYftR .+ cuYffI .* cuYftI; Yimfrom .= .- cuYffR .* cuYftI .+ cuYffI .* cuYftR
  Yreto   .= cuYtfR .* cuYttR .+ cuYtfI .* cuYttI; Yimto   .= .- cuYtfR .* cuYttI .+ cuYtfI .* cuYttR
  @show mconnect = maximum(size.(cuFromLines,1))
  # viewVmToFromLines = CuArray{ForwardDiff.Dual{Nothing,Float64,3}, 2, Nothing}(undef, nbus, mconnect)
  viewVmToFromLines   = Array{CuArray,1}(undef, nbus)
  viewcuYftRFromLines = Array{CuArray,1}(undef, nbus)
  viewVaToFromLines   = Array{CuArray,1}(undef, nbus)
  viewcuYftIFromLines = Array{CuArray,1}(undef, nbus)
  # viewVmToFromLines = Array{SubArray,1}(undef, nbus)
  # viewcuYftRFromLines = Array{SubArray,1}(undef, nbus)
  # viewVaToFromLines = Array{SubArray,1}(undef, nbus)
  # viewcuYftIFromLines = Array{SubArray,1}(undef, nbus)
  
  function filltopo(array,range)
    for i in range
      @show i 
    end
  end
  # for b in 1:nbus @show size(cuVm[mapbus2lineto[cuFromLines[b]]]) end
  # for b in 1:nbus @show size(viewVmToFromLines[b,:]) end
  # for b in 1:nbus @show viewVmToFromLines[b,:] end
  # for b in 1:nbus @show cuVm[mapbus2lineto[cuFromLines[b]]] end
  # for b in 1:nbus filltopo(viewVmToFromLines[b,:], mapbus2lineto[cuFromLines[b]]) end
  for b in 1:nbus viewVmToFromLines[b]   = cuVm[mapbus2lineto[cuFromLines[b]]] end
  for b in 1:nbus viewcuYftRFromLines[b] = cuYftR[cuFromLines[b]] end 
  for b in 1:nbus viewVaToFromLines[b]   = cuVa[mapbus2lineto[cuFromLines[b]]] end
  for b in 1:nbus viewcuYftIFromLines[b] = cuYftI[cuFromLines[b]] end

  # for b in 1:nbus viewVmToFromLines[b]   = view(viewVmTo, cuFromLines[b]) end
  # for b in 1:nbus viewcuYftRFromLines[b] = view(cuYftR, cuFromLines[b]) end 
  # for b in 1:nbus viewVaToFromLines[b]   = view(viewVaTo, cuFromLines[b]) end
  # for b in 1:nbus viewcuYftIFromLines[b] = view(cuYftI, cuFromLines[b]) end

  # for b in 1:nbus @show cuVm[mapbus2lineto[cuFromLines[b]]] end
  # for b in 1:nbus @show viewVmToFromLines[b] end

  viewVmFromToLines = Array{CuArray,1}(undef, nbus)
  viewcuYtfRToLines = Array{CuArray,1}(undef, nbus)
  viewVaFromToLines = Array{CuArray,1}(undef, nbus)
  viewcuYtfIToLines = Array{CuArray,1}(undef, nbus)
  for b in 1:nbus viewVmFromToLines[b] = cuVm[mapbus2linefrom[cuToLines[b]]] end
  for b in 1:nbus viewcuYtfRToLines[b] = cuYtfR[cuToLines[b]] end 
  for b in 1:nbus viewVaFromToLines[b] = cuVa[mapbus2linefrom[cuToLines[b]]] end
  for b in 1:nbus viewcuYtfIToLines[b] = cuYtfI[cuToLines[b]] end

  # viewVmToFromLines = Array{SubArray,1}(undef, nbus)
  # viewVaFromFromLines = Array{SubArray,1}(undef, nbus)
  # for b in 1:nbus viewVmToFromLines[b]   = view(viewVmTo, FromLines[b]) end
  # for b in 1:nbus viewcuYftRFromLines[b] = view(cuYftR, FromLines[b]) end 
  # for b in 1:nbus viewVaToFromLines[b]   = view(viewVaTo, FromLines[b]) end
  # for b in 1:nbus viewcuYftIFromLines[b] = view(cuYftI, FromLines[b]) end

  # viewVmFromToLines = Array{SubArray,1}(undef, nbus)
  # viewcuYtfRToLines = Array{SubArray,1}(undef, nbus)
  # viewVaFromToLines = Array{SubArray,1}(undef, nbus)
  # viewcuYtfIToLines = Array{SubArray,1}(undef, nbus)
  # for b in 1:nbus 
  #   arrays.viewFromR[b] = sum(arrays.cuVm[b] .* view(arrays.viewVmFrom, FromLines[b]) .* (view(arrays.cuYtfR, FromLines[b]) .* CUDAnative.cos.(arrays.cuVa[b] .- view(arrays.viewVaFrom, FromLines[b])) .+ view(arrays.cuYtfI, FromLines[b]).* CUDAnative.sin.(arrays.cuVa[b] .- view(arrays.viewVaFrom, FromLines[b])))) 
  # end
  # for b in 1:nbus 
  #   arrays.viewToI[b] = sum(arrays.cuVm[b] .* view(arrays.viewVmTo, FromLines[b]) .* (.- view(arrays.cuYftI, FromLines[b]) .* CUDAnative.cos.(arrays.cuVa[b] .- view(arrays.viewVaTo, FromLines[b])) .+ view(arrays.cuYftR, FromLines[b]).* CUDAnative.sin.(arrays.cuVa[b] .- view(arrays.viewVaTo, FromLines[b])))) 
  # end
  # for b in 1:nbus 
  #   arrays.viewFromI[b] = sum(arrays.cuVm[b] .* view(arrays.viewVmFrom, FromLines[b]) .* (.- view(arrays.cuYtfI, FromLines[b]) .* CUDAnative.cos.(arrays.cuVa[b] .- view(arrays.viewVaFrom, FromLines[b])) .+ view(arrays.cuYtfR, FromLines[b]).* CUDAnative.sin.(arrays.cuVa[b] .- view(arrays.viewVaFrom, FromLines[b])))) 
  # end
  return CompArrays(cuPg, cuQg, cuVa, cuVm, baseMVA, nbus, nline, ngen, 
                    coeff0, coeff1, coeff2, # balance constraints
                    viewToR, viewFromR, viewToI, viewFromI,
                    viewVmTo, viewVmFrom, viewVaTo, viewVaFrom,
                    viewYffR, viewYttR, viewYffI, viewYttI,
                    cuYffR, cuYffI, cuYttR, cuYttI, cuYftR, cuYftI, cuYtfR, cuYtfI, cuYshR, cuYshI,
                    viewPg, viewQg,
                    cuPd, cuQd,
                    cuflowmax,
                    Yff_abs2, 
                    Yft_abs2,
                    Ytf_abs2,
                    Ytt_abs2,
                    Yrefrom,
                    Yimfrom,
                    Yreto,
                    Yimto,
                    viewVmToFromLines, viewcuYftRFromLines, viewVaToFromLines, viewcuYftIFromLines,
                    viewVmFromToLines, viewcuYtfRToLines, viewVaFromToLines, viewcuYtfIToLines)
end

function objective(opf_data::OPFData, arrays::CompArrays) where T
  # minimize active power
  return sum(arrays.coeff2 .* (arrays.baseMVA .* arrays.cuPg).^2 
    .+ arrays.coeff1 .* (arrays.baseMVA .* arrays.cuPg)
    .+ arrays.coeff0)
end

function constraints(rbalconst::T, ibalconst::T, limitsto, limitsfrom, opf_data, arrays, timeroutput) where T
  lines = opf_data.lines; buses = opf_data.buses; generators = opf_data.generators; baseMVA = opf_data.baseMVA
  busIdx = opf_data.BusIdx; FromLines = opf_data.FromLines; ToLines = opf_data.ToLines; BusGeners = opf_data.BusGenerators;

  nbus  = length(buses); nline = length(lines); ngen  = length(generators)


  @timeit timeroutput "constraints build arrays" begin
  # *₊(a,b) = a .* b 
  # -₊(a,b) = a .* b 
  # function cos₊(a)
  #   return CUDAnative.cos.(a) 
  # end
  # arraycuVm = Array{CuArray,1}(undef,nbus)
  # for b in 1:nbus arraycuVm[b] = arrays.cuVm[b] end
  # arraycuVa = Array{CuArray,1}(undef,nbus)
  # for b in 1:nbus arraycuVa[b] = arrays.cuVa[b] end
  # @show typeof(arrays.viewVmToFromLines)
  # @show typeof(arrays.viewVaToFromLines)
  # @show typeof(arrays.viewVmToFromLines .*₊ arrays.viewcuYftRFromLines)
  # @show typeof(arraycuVm .*₊ arrays.viewcuYftRFromLines)
  # @show size(arrays.viewVmToFromLines .*₊ arrays.viewcuYftRFromLines)
  # @show typeof(.-₊(arraycuVa, arrays.viewVaToFromLines))
  # @show typeof(arrays.cuVa[1] .- arrays.viewVaToFromLines[1])
  
  # # @show arrays.cuVa .(-₊) arrays.viewVaToFromLines
  # cos₊(arraycuVa .-₊ arrays.viewVaToFromLines)
   
  for b in 1:nbus 
    arrays.viewToR[b] = sum(arrays.cuVm[b] .* arrays.viewVmToFromLines[b] .* (arrays.viewcuYftRFromLines[b] .* CUDAnative.cos.(arrays.cuVa[b] .- arrays.viewVaToFromLines[b]) .+ arrays.viewcuYftIFromLines[b] .* CUDAnative.sin.(arrays.cuVa[b] .- arrays.viewVaToFromLines[b]))) 
  # end
  # for b in 1:nbus 
    arrays.viewFromR[b] = sum(arrays.cuVm[b] .* arrays.viewVmFromToLines[b] .* (arrays.viewcuYtfRToLines[b] .* CUDAnative.cos.(arrays.cuVa[b] .- arrays.viewVaFromToLines[b]) .+ arrays.viewcuYtfIToLines[b] .* CUDAnative.sin.(arrays.cuVa[b] .- arrays.viewVaFromToLines[b]))) 
  end
  for b in 1:nbus 
    arrays.viewToI[b] = sum(arrays.cuVm[b] .* arrays.viewVmToFromLines[b] .* (.- arrays.viewcuYftIFromLines[b] .* CUDAnative.cos.(arrays.cuVa[b] .- arrays.viewVaToFromLines[b]) .+ arrays.viewcuYftRFromLines[b] .* CUDAnative.sin.(arrays.cuVa[b] .- arrays.viewVaToFromLines[b]))) 
  end
  for b in 1:nbus 
    arrays.viewFromI[b] = sum(arrays.cuVm[b] .* arrays.viewVmFromToLines[b] .* (.- arrays.viewcuYtfIToLines[b] .* CUDAnative.cos.(arrays.cuVa[b] .- arrays.viewVaFromToLines[b]) .+ arrays.viewcuYtfRToLines[b] .* CUDAnative.sin.(arrays.cuVa[b] .- arrays.viewVaFromToLines[b]))) 
  end
  end
  @timeit timeroutput "balance constraints" begin
  rbalconst .= (((arrays.viewYffR .+ arrays.viewYttR) .+ arrays.cuYshR) .* arrays.cuVm.^2) .+ arrays.viewToR .+ arrays.viewFromR .- ((arrays.viewPg .* baseMVA) .- arrays.cuPd) ./ baseMVA 

  ibalconst .= (((arrays.viewYffI .+ arrays.viewYttI) .- arrays.cuYshI) .* arrays.cuVm.^2) .+ arrays.viewToI .+ arrays.viewFromI .- ((arrays.viewQg .* baseMVA) .- arrays.cuQd) ./ baseMVA 
  end

  # branch apparent power limits (from bus)
  @timeit timeroutput "line constraints" begin
  limitsto .= (arrays.viewVmFrom.^2 .* (arrays.Yff_abs2 .* arrays.viewVmFrom.^2 .+ arrays.Yft_abs2 .* arrays.viewVmTo.^2
           .+ 2 .* arrays.viewVmFrom .* arrays.viewVmTo .* (arrays.Yrefrom .* CUDAnative.cos.(arrays.viewVaFrom .- arrays.viewVaTo) .- arrays.Yimfrom .* CUDAnative.sin.(arrays.viewVaFrom .- arrays.viewVaTo)))
           .- arrays.cuflowmax)
  # branch apparent power limits (to bus)
  limitsfrom .= (arrays.viewVmTo.^2 .* (arrays.Ytf_abs2 .* arrays.viewVmFrom.^2 .+ arrays.Ytt_abs2 .* arrays.viewVmTo.^2
             .+ 2 .* arrays.viewVmFrom .* arrays.viewVmTo .* (arrays.Yreto .* CUDAnative.cos.(arrays.viewVaFrom - arrays.viewVaTo) .- arrays.Yimto .* CUDAnative.sin.(arrays.viewVaFrom .- arrays.viewVaTo)))
             .- arrays.cuflowmax)
  end
  return
end

function acopf_outputAll(opfmodel, opf_data, Pg, Qg, Va, Vm)
  #shortcuts for compactness
  lines = opf_data.lines; buses = opf_data.buses; generators = opf_data.generators; baseMVA = opf_data.baseMVA
  busIdx = opf_data.BusIdx; FromLines = opf_data.FromLines; ToLines = opf_data.ToLines; BusGeners = opf_data.BusGenerators;

  nbus  = length(buses); nline = length(lines); ngen  = length(generators)

  # OUTPUTING
  println("Objective value: ", objective_value(opfmodel), "USD/hr")
  VM = Array{Float64}(undef, size(Vm, 1))
  VM=value.(Vm); VA=value.(Va)
  PG=value.(Pg); QG=value.(Qg)

  println("============================= BUSES ==================================")
  println("  BUS    Vm     Va   |   Pg (MW)    Qg(MVAr) ")   # |    P (MW)     Q (MVAr)")  #|         (load)   ") 
  
  println("                     |     (generation)      ") 
  println("----------------------------------------------------------------------")
  for i in 1:nbus
    @printf("%4d | %6.2f  %6.2f | %s  | \n",
	    buses[i].bus_i, VM[i], VA[i]*180/pi, 
	    length(BusGeners[i])==0 ? "   --          --  " : @sprintf("%7.2f     %7.2f", baseMVA*PG[BusGeners[i][1]], baseMVA*QG[BusGeners[i][1]]))
  end   
  println("\n")

  within=20 # percentage close to the limits
  
  
  nflowlim=0
  for l in 1:nline
    if lines[l].rateA!=0 && lines[l].rateA<1.0e10
      nflowlim += 1
    end
  end

  if nflowlim>0 
    println("Number of lines with flow limits: ", nflowlim)

    optvec=zeros(2*nbus+2*ngen)
    optvec[1:ngen]=PG
    optvec[ngen+1:2*ngen]=QG
    optvec[2*ngen+1:2*ngen+nbus]=VM
    optvec[2*ngen+nbus+1:2*ngen+2*nbus]=VA

    d = JuMP.NLPEvaluator(opfmodel)
    MOI.initialize(d, [:Jac])

    consRhs = zeros(2*nbus+2*nflowlim)
    MOI.eval_constraint(d, consRhs, optvec)  


    #println(consRhs)

    @printf("================ Lines within %d %s of flow capacity ===================\n", within, "%")
    println("Line   From Bus    To Bus    At capacity")

    nlim=1
    for l in 1:nline
      if lines[l].rateA!=0 && lines[l].rateA<1.0e10
        flowmax=(lines[l].rateA/baseMVA)^2
        idx = 2*nbus+nlim
        
        if( (consRhs[idx]+flowmax)  >= (1-within/100)^2*flowmax )
          @printf("%3d      %3d      %3d        %5.3f%s\n", l, lines[l].from, lines[l].to, 100*sqrt((consRhs[idx]+flowmax)/flowmax), "%" ) 
          # @printf("%7.4f   %7.4f    %7.4f \n", consRhs[idx], consRhs[idx]+flowmax,  flowmax)
        end
        nlim += 1
      end
    end
  end

  #println(getvalue(Vm))
  #println(getvalue(Va)*180/pi)

  #println(getvalue(Pg))
  #println(getvalue(Qg))

  return
end


# Compute initial point for IPOPT based on the values provided in the case data
function acopf_initialPt_IPOPT(opfdata)
  Pg=zeros(length(opfdata.generators)); Qg=zeros(length(opfdata.generators)); i=1
  for g in opfdata.generators
    # set the power levels in in between the bounds as suggested by matpower 
    # (case data also contains initial values in .Pg and .Qg - not used with IPOPT)
    Pg[i]=0.5*(g.Pmax+g.Pmin)
    Qg[i]=0.5*(g.Qmax+g.Qmin)
    i=i+1
  end
  @assert i-1==length(opfdata.generators)

  Vm=zeros(length(opfdata.buses)); i=1;
  for b in opfdata.buses
    # set the ini val for voltage magnitude in between the bounds 
    # (case data contains initials values in Vm - not used with IPOPT)
    Vm[i]=0.5*(b.Vmax+b.Vmin); 
    i=i+1
  end
  @assert i-1==length(opfdata.buses)

  # set all angles to the angle of the reference bus
  Va = opfdata.buses[opfdata.bus_ref].Va * ones(length(opfdata.buses))

  return Pg,Qg,Vm,Va
end
function myseed!(duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
               seeds::AbstractArray{ForwardDiff.Partials{N,V}}, timeroutput) where {T,V,N}

  @timeit timeroutput "myseed 1" begin
    for i in 1:N
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
    end
  end
  @timeit timeroutput "myseed 2" begin
    zeroseed::ForwardDiff.Partials{N,V} = zero(ForwardDiff.Partials{N,V}) 
    for i in N+1:size(duals,1)
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], zeroseed)
    end
  end
    return duals
end
function benchmark(opfdata, Pg, Qg, Vm, Va, npartials, mpartials, loops, timeroutput)
  
  t1s{N} =  ForwardDiff.Dual{Nothing,Float64, N} where N
  t2s{M,N} =  ForwardDiff.Dual{Nothing,t1s{N}, M} where {N, M}

  T = CuArray{Float64,1,Nothing}

  t1sseedvec = zeros(Float64, npartials)
  t1sseeds = Array{ForwardDiff.Partials{npartials,Float64},1}(undef, size(Pg,1))
  for i in 1:npartials
    t1sseedvec[i] = 1.0
    t1sseeds[i] = ForwardDiff.Partials{npartials, Float64}(NTuple{npartials, Float64}(t1sseedvec))
    t1sseedvec[i] = 0.0
  end
  for i in npartials+1:size(Pg,1)
    t1sseeds[i] = ForwardDiff.Partials{npartials, Float64}(NTuple{npartials, Float64}(t1sseedvec))
  end

  @timeit timeroutput "t1s seeding" begin
  t1scuPg = acopf.myseed!(CuArray{t1s{npartials}, 1, Nothing}(undef, size(Pg,1)), value.(Pg), t1sseeds, timeroutput)
  t1scuQg = ForwardDiff.seed!(CuArray{t1s{npartials}, 1, Nothing}(undef, size(Qg,1)), value.(Qg))
  t1scuVa = ForwardDiff.seed!(CuArray{t1s{npartials}, 1, Nothing}(undef, size(Va,1)), value.(Va))
  t1scuVm = ForwardDiff.seed!(CuArray{t1s{npartials}, 1, Nothing}(undef, size(Vm,1)), value.(Vm))
  end
  # @show t1scuPg


  t2sseedvec = Array{t1s{npartials},1}(undef, mpartials)
  t2sseeds = Array{ForwardDiff.Partials{mpartials,t1s{npartials}},1}(undef, size(Pg,1))
  t2sseedvec .= 0.0
  for i in 1:mpartials
    t2sseedvec[i] = 1.0
    t2sseeds[i] = ForwardDiff.Partials{mpartials, t1s{npartials}}(NTuple{mpartials, t1s{npartials}}(t2sseedvec))
    t2sseedvec[i] = 0.0
  end
  for i in mpartials+1:size(Pg,1)
    t2sseeds[i] = ForwardDiff.Partials{mpartials, t1s{npartials}}(NTuple{mpartials, t1s{npartials}}(t2sseedvec))
  end
  # @show t2sseeds
  # @show t2sseedvec
  # t2sseedtup = NTuple{size(Pg,1), ForwardDiff.Partials{size(Pg,1), Float64}}(t1sseeds)
  @timeit timeroutput "t2s seeding" begin
  t2scuPg = acopf.myseed!(CuArray{t2s{mpartials,npartials}, 1, Nothing}(undef, size(Pg,1)), t1scuPg, t2sseeds, timeroutput)
  # t2scuPg = ForwardDiff.seed!(CuArray{t2s{mpartials, npartials}, 1, Nothing}(undef, size(Pg,1)), t1scuPg)
  t2scuQg = ForwardDiff.seed!(CuArray{t2s{mpartials, npartials}, 1, Nothing}(undef, size(Qg,1)), t1scuQg)
  t2scuVa = ForwardDiff.seed!(CuArray{t2s{mpartials, npartials}, 1, Nothing}(undef, size(Va,1)), t1scuVa)
  t2scuVm = ForwardDiff.seed!(CuArray{t2s{mpartials, npartials}, 1, Nothing}(undef, size(Vm,1)), t1scuVm)
  end
  # @show t2scuPg

  # Pg0,Qg0,Vm0,Va0 = acopf.acopf_initialPt_IPOPT(opfdata)
  # cuPg = ForwardDiff.seed!(CuArray{ForwardDiff.Dual{Nothing,Float64,size(Pg,1)}, 1, Nothing}(undef, size(Pg,1)), Pg0, seedtup)
  # cuQg = ForwardDiff.seed!(CuArray{ForwardDiff.Dual{Nothing,Float64,size(Pg,1)}, 1, Nothing}(undef, size(Qg,1)), Qg0)
  # cuVa = ForwardDiff.seed!(CuArray{ForwardDiff.Dual{Nothing,Float64,size(Pg,1)}, 1, Nothing}(undef, size(Va,1)), Va0)
  # cuVm = ForwardDiff.seed!(CuArray{ForwardDiff.Dual{Nothing,Float64,size(Pg,1)}, 1, Nothing}(undef, size(Vm,1)), Vm0)
  T = typeof(t1scuQg)
  t1srbalconst = T(undef, length(opfdata.buses))
  t1sibalconst = T(undef, length(opfdata.buses))
  t1slimitsto = T(undef, length(opfdata.lines))
  t1slimitsfrom = T(undef, length(opfdata.lines))
  println("Create t1s arrays")
  @timeit timeroutput "Create t1s arrays" begin
  t1sarrays = acopf.create_arrays(t1scuPg, t1scuQg, t1scuVa, t1scuVm, opfdata)
  end
  println("Initial t1s objective")
  @timeit timeroutput "Initial t1s objective" begin
  t1sPg = acopf.objective(opfdata, t1sarrays)
  end
  println("t1s objective")
  @timeit timeroutput "t1s objective" begin
    for i in 1:loops
      t1sPg = acopf.objective(opfdata, t1sarrays)
    end
  end
  println("Initial t1s constraints")
  @timeit timeroutput "Initial t1s constraints" begin
  acopf.constraints(t1srbalconst, t1sibalconst, t1slimitsto, t1slimitsfrom, opfdata, t1sarrays, timeroutput)
  end
  println("t1s constraints")
  @timeit timeroutput "t1s constraints" begin
    for i in 1:loops
      acopf.constraints(t1srbalconst, t1sibalconst, t1slimitsto, t1slimitsfrom, opfdata, t1sarrays, timeroutput)
    end
  end
  T = typeof(t2scuQg)
  t2srbalconst = T(undef, length(opfdata.buses))
  t2sibalconst = T(undef, length(opfdata.buses))
  t2slimitsto = T(undef, length(opfdata.lines))
  t2slimitsfrom = T(undef, length(opfdata.lines))
  println("Create t2s arrays")
  @timeit timeroutput "Create t2s arrays" begin
  t2sarrays = acopf.create_arrays(t2scuPg, t2scuQg, t2scuVa, t2scuVm, opfdata)
  end
  println("Initial t2s objective")
  @timeit timeroutput "Initial t2s objective" begin
  t2sPg = acopf.objective(opfdata, t2sarrays)
  end
  println("t2s objective")
  @timeit timeroutput "t2s objective" begin
    for i in 1:loops
      t2sPg = acopf.objective(opfdata, t2sarrays)
    end
  end
  println("Initial t2s constraints")
  @timeit timeroutput "Initial t2s constraints" begin
  acopf.constraints(t2srbalconst, t2sibalconst, t2slimitsto, t2slimitsfrom, opfdata, t1sarrays, timeroutput)
  end
  println("t2s constraints")
  @timeit timeroutput "t2s constraints" begin
    for i in 1:loops
      acopf.constraints(t2srbalconst, t2sibalconst, t2slimitsto, t2slimitsfrom, opfdata, t1sarrays, timeroutput)
    end
  end
  return t1sPg, t2sPg
end
end

