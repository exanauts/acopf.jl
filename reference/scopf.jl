using JuMP
using Ipopt
#using Hiop
using Printf
using DelimitedFiles
using SparseArrays
using acopf

function solve(opfmodel)
  optimize!(opfmodel)
  status = termination_status(opfmodel)
  if status != MOI.LOCALLY_SOLVED
      println("Could not solve the model to optimality.")
  end

  return opfmodel, status
end

function auglag_model(opfdata; max_iter=100, rho=10)
  Pg0,Qg0,Vm0,Va0 = acopf.initialPt_IPOPT(opfdata)
  lines = opfdata.lines; buses = opfdata.buses; generators = opfdata.generators; baseMVA = opfdata.baseMVA
  busIdx = opfdata.BusIdx; FromLines = opfdata.FromLines; ToLines = opfdata.ToLines; BusGeners = opfdata.BusGenerators;
  nbus=length(opfdata.buses); ngen=length(opfdata.generators)

  generators=opfdata.generators; buses=opfdata.buses; baseMVA = opfdata.baseMVA

  opfmodel = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => max_iter))

  ncont = 1

  @variable(opfmodel, generators[i].Pmin <= Pg[i=1:ngen] <= generators[i].Pmax, start = Pg0[i])
  @variable(opfmodel, 0<=extra[i=1:ngen,0:ncont]<=0.05*generators[i].Pmax, start = 0.025*generators[i].Pmax)
  @variable(opfmodel, generators[i].Qmin <= Qg[i=1:ngen] <= generators[i].Qmax, start = Qg0[i])

  @variable(opfmodel, buses[i].Vmin <= Vm[i=1:nbus, c=0:ncont] <= buses[i].Vmax, start = Vm0[i])
  @variable(opfmodel, Va[i=1:nbus, c=0:ncont], start = Va0[i])

  #fix the voltage angle at the reference bus
  for c in 0:ncont
    set_lower_bound(Va[opfdata.bus_ref,c], buses[opfdata.bus_ref].Va)
    set_upper_bound(Va[opfdata.bus_ref,c], buses[opfdata.bus_ref].Va)
  end

  @variable(opfmodel, generators[g].Pmin <= Pgc[g=1:ngen,co=0:ncont] <= generators[g].Pmax)
  @NLexpression(opfmodel, post_gen[g=1:ngen,co=0:ncont], Pgc[g,co] - (Pg[g] + extra[g,co]))

  #@constraint(opfmodel, ex[i=1:ngen,co=0:ncont], generators[i].Pmin <= Pg[i] + extra[i,co] <= generators[i].Pmax)
  #@NLexpression(opfmodel, Pgc[g=1:ngen, co=0:ncont], Pg[g] + extra[g,co])
  @NLexpression(opfmodel, Qgc[g=1:ngen, co=0:ncont], Qg[g])
  zeroexpr = @NLexpression(opfmodel, 0)
  set_lower_bound(Pgc[1,1], 0)
  set_upper_bound(Pgc[1,1], 0)
  # Pgc[1,1] = zeroexpr
  # Pgc[2,2] = zeroexpr
  # Pgc[3,3] = zeroexpr
  Qgc[1,1] = zeroexpr
  # Qgc[2,2] = zeroexpr
  # Qgc[3,3] = zeroexpr
  # @show Pgc

  @NLexpression(opfmodel, objective, sum( generators[i].coeff[generators[i].n-2]*(baseMVA*(Pgc[i,c]))^2
			             +generators[i].coeff[generators[i].n-1]*(baseMVA*(Pgc[i,c]))
				     +generators[i].coeff[generators[i].n  ] for i=1:ngen, c=0:ncont) / (1+ncont))

  YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(opfdata.lines, opfdata.buses, opfdata.baseMVA)
  lines=opfdata.lines
  busIdx=opfdata.BusIdx;FromLines=opfdata.FromLines; ToLines=opfdata.ToLines; BusGeners=opfdata.BusGenerators
  nline=length(lines)

  # power flow balance
  @NLexpression( opfmodel, real_power[b=1:nbus,co=0:ncont],
        ( sum( YffR[l] for l in FromLines[b]) + sum( YttR[l] for l in ToLines[b]) + YshR[b] ) * Vm[b,co]^2
        + sum(  Vm[b,co]*Vm[busIdx[lines[l].to],  co]*( YftR[l]*cos(Va[b,co]-Va[busIdx[lines[l].to],  co])
              + YftI[l]*sin(Va[b,co]-Va[busIdx[lines[l].to],co]  )) for l in FromLines[b] )
        + sum(  Vm[b,co]*Vm[busIdx[lines[l].from],co]*( YtfR[l]*cos(Va[b,co]-Va[busIdx[lines[l].from],co])
              + YtfI[l]*sin(Va[b,co]-Va[busIdx[lines[l].from],co])) for l in ToLines[b]   )
        -(sum(baseMVA*Pgc[g,co] for g in BusGeners[b]) - buses[b].Pd) / baseMVA      # Sbus part
        )

  @NLexpression( opfmodel, reactive_power[b=1:nbus,co=0:ncont],
        ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[b,co]^2
        + sum(  Vm[b,co]*Vm[busIdx[lines[l].to],co]  *(-YftI[l]*cos(Va[b,co]-Va[busIdx[lines[l].to],  co])
              + YftR[l]*sin(Va[b,co]-Va[busIdx[lines[l].to],co]  )) for l in FromLines[b] )
        + sum( Vm[b,co]*Vm[busIdx[lines[l].from],co] *(-YtfI[l]*cos(Va[b,co]-Va[busIdx[lines[l].from],co])
              + YtfR[l]*sin(Va[b,co]-Va[busIdx[lines[l].from],co])) for l in ToLines[b]   )
        -(sum(baseMVA*Qgc[g] for g in BusGeners[b]) - buses[b].Qd) / baseMVA      #Sbus part
       )

  # branch/lines flow limits
  limidx = findall(x -> x.rateA != 0 && x.rateA < 1.0e10, lines)
  nlinelim = length(limidx)
  Yff_abs2 = YffR.^2 .+ YffI.^2
  Yft_abs2 = YftR.^2 .+ YftI.^2
  Yre_fr   =  YffR.*YftR .+ YffI.*YftI
  Yim_fr   = -YffR.*YftI .+ YffI.*YftR
  flowmax = [(lines[l].rateA / baseMVA)^2 for l in limidx]

  @variable(opfmodel, Sfrom[l=1:nlinelim] >= 0, start=1e4)
  @variable(opfmodel, Sto[l=1:nlinelim] >= 0, start=1e4)

  @NLexpression(opfmodel, powerlimit_from[l=1:nlinelim,co=0:ncont],
    Vm[busIdx[lines[limidx[l]].from],co]^2 *
    ( Yff_abs2[limidx[l]]*Vm[busIdx[lines[limidx[l]].from],co]^2 + Yft_abs2[limidx[l]]*Vm[busIdx[lines[limidx[l]].to],co]^2
    + 2*Vm[busIdx[lines[limidx[l]].from],co]*Vm[busIdx[lines[limidx[l]].to],co]
              *(Yre_fr[limidx[l]]*cos(Va[busIdx[lines[limidx[l]].from],co]-Va[busIdx[lines[limidx[l]].to],co])-
                Yim_fr[limidx[l]]*sin(Va[busIdx[lines[limidx[l]].from],co]-Va[busIdx[lines[limidx[l]].to],co]))
    ) - flowmax[l] + Sfrom[l]
  )

  Ytf_abs2 = YtfR.^2 .+ YtfI.^2
  Ytt_abs2 = YttR.^2 .+ YttI.^2
  Yre_to   =  YtfR.*YttR .+ YtfI.*YttI
  Yim_to   = -YtfR.*YttI .+ YtfI.*YttR

  @NLexpression(opfmodel, powerlimit_to[l=1:nlinelim,co=0:ncont],
    Vm[busIdx[lines[limidx[l]].to],co]^2 *
    ( Ytf_abs2[limidx[l]]*Vm[busIdx[lines[limidx[l]].from],co]^2 + Ytt_abs2[limidx[l]]*Vm[busIdx[lines[limidx[l]].to],co]^2
    + 2*Vm[busIdx[lines[limidx[l]].from],co]*Vm[busIdx[lines[limidx[l]].to],co]
        *(Yre_to[limidx[l]]*cos(Va[busIdx[lines[limidx[l]].from],co]-Va[busIdx[lines[limidx[l]].to],co])-
          Yim_to[limidx[l]]*sin(Va[busIdx[lines[limidx[l]].from],co]-Va[busIdx[lines[limidx[l]].to],co]))
    ) - flowmax[l] + Sto[l]
  )

  # minimize active power
#  @NLobjective(opfmodel,
#		  Min,
#		  sum( generators[i].coeff[generators[i].n] +
#		       sum(generators[i].coeff[generators[i].n-k]*(baseMVA*Pg[i])^k for k=1:generators[i].n-1)
#		       for i=1:ngen)
#		 )


  lambda_post_pg = ones((ngen,ncont+1))
  lambda_pg = ones((nbus,ncont+1))
  lambda_qg = ones((nbus,ncont+1))
  lambda_sfrom = ones((nlinelim,ncont+1))
  lambda_sto = ones((nlinelim,ncont+1))

  @NLobjective(opfmodel, Min,
               objective +
               sum(lambda_post_pg[g,co+1]*post_gen[g,co] for g=1:ngen,co=0:ncont) +
               sum(lambda_pg[b,co+1]*real_power[b,co] for b=1:nbus,co=0:ncont) +
               sum(lambda_qg[b,co+1]*reactive_power[b,co] for b=1:nbus,co=0:ncont) +
               sum(lambda_sfrom[l,co+1]*powerlimit_from[l,co] for l=1:nlinelim,co=0:ncont) +
               sum(lambda_sto[l,co+1]*powerlimit_to[l,co] for l=1:nlinelim,co=0:ncont) +
               (rho/2)*sum((post_gen[g,co])^2 for g=1:ngen,co=0:ncont) +
               (rho/2)*sum((real_power[b,co])^2 for b=1:nbus,co=0:ncont) +
               (rho/2)*sum((reactive_power[b,co])^2 for b=1:nbus,co=0:ncont) +
               (rho/2)*sum((powerlimit_from[l,co])^2 for l=1:nlinelim,co=0:ncont) +
               (rho/2)*sum((powerlimit_to[l,co])^2 for l=1:nlinelim,co=0:ncont)
               )

  return opfmodel
end

function model(opfdata; max_iter=100)
  Pg0,Qg0,Vm0,Va0 = acopf.initialPt_IPOPT(opfdata)
  lines = opfdata.lines; buses = opfdata.buses; generators = opfdata.generators; baseMVA = opfdata.baseMVA
  busIdx = opfdata.BusIdx; FromLines = opfdata.FromLines; ToLines = opfdata.ToLines; BusGeners = opfdata.BusGenerators;
  nbus=length(opfdata.buses); ngen=length(opfdata.generators)

  generators=opfdata.generators; buses=opfdata.buses; baseMVA = opfdata.baseMVA

  opfmodel = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => max_iter))

  ncont = 1

  @variable(opfmodel, generators[i].Pmin <= Pg[i=1:ngen] <= generators[i].Pmax, start = Pg0[i])
  @variable(opfmodel, 0<=extra[i=1:ngen,0:ncont]<=0.05*generators[i].Pmax, start = 0.025*generators[i].Pmax)
  @variable(opfmodel, generators[i].Qmin <= Qg[i=1:ngen] <= generators[i].Qmax, start = Qg0[i])



  @variable(opfmodel, buses[i].Vmin <= Vm[i=1:nbus, c=0:ncont] <= buses[i].Vmax, start = Vm0[i])
  @variable(opfmodel, Va[i=1:nbus, c=0:ncont], start = Va0[i])

  #fix the voltage angle at the reference bus
  for c in 0:ncont
    set_lower_bound(Va[opfdata.bus_ref,c], buses[opfdata.bus_ref].Va)
    set_upper_bound(Va[opfdata.bus_ref,c], buses[opfdata.bus_ref].Va)
  end

  @constraint(opfmodel, ex[i=1:ngen,co=0:ncont], generators[i].Pmin <= Pg[i] + extra[i,co] <= generators[i].Pmax)
  @NLexpression(opfmodel, Pgc[g=1:ngen, co=0:ncont], Pg[g] + extra[g,co])
  @NLexpression(opfmodel, Qgc[g=1:ngen, co=0:ncont], Qg[g])
  zeroexpr = @NLexpression(opfmodel, 0)
  Pgc[1,1] = zeroexpr
  # Pgc[2,2] = zeroexpr
  # Pgc[3,3] = zeroexpr
  Qgc[1,1] = zeroexpr
  # Qgc[2,2] = zeroexpr
  # Qgc[3,3] = zeroexpr
  # @show Pgc

  @NLobjective(opfmodel, Min, sum( generators[i].coeff[generators[i].n-2]*(baseMVA*(Pgc[i,c]))^2
			             +generators[i].coeff[generators[i].n-1]*(baseMVA*(Pgc[i,c]))
				     +generators[i].coeff[generators[i].n  ] for i=1:ngen, c=0:ncont) / (1+ncont))

  for co=0:ncont
    #branch admitances
    # if co==0
      YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(opfdata.lines, opfdata.buses, opfdata.baseMVA)
      lines=opfdata.lines
      busIdx=opfdata.BusIdx;FromLines=opfdata.FromLines; ToLines=opfdata.ToLines; BusGeners=opfdata.BusGenerators
    # else
    #   opfm1=opf_loaddata(ARGS[1], sd.lines_off[co])
    #   YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(opfm1.lines, opfm1.buses, opfm1.baseMVA)
    #   lines=opfm1.lines
    #   busIdx = opfm1.BusIdx; FromLines = opfm1.FromLines; ToLines = opfm1.ToLines; BusGeners = opfm1.BusGenerators
    # end
    nline=length(lines)


    # power flow balance
    for b in 1:nbus
      #real part
      @NLconstraint( opfmodel,
        ( sum( YffR[l] for l in FromLines[b]) + sum( YttR[l] for l in ToLines[b]) + YshR[b] ) * Vm[b,co]^2
        + sum(  Vm[b,co]*Vm[busIdx[lines[l].to],  co]*( YftR[l]*cos(Va[b,co]-Va[busIdx[lines[l].to],  co])
              + YftI[l]*sin(Va[b,co]-Va[busIdx[lines[l].to],co]  )) for l in FromLines[b] )
        + sum(  Vm[b,co]*Vm[busIdx[lines[l].from],co]*( YtfR[l]*cos(Va[b,co]-Va[busIdx[lines[l].from],co])
              + YtfI[l]*sin(Va[b,co]-Va[busIdx[lines[l].from],co])) for l in ToLines[b]   )
        -(sum(baseMVA*Pgc[g,co] for g in BusGeners[b]) - buses[b].Pd) / baseMVA      # Sbus part
        ==0)

      #imaginary part
      @NLconstraint( opfmodel,
        ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[b,co]^2
        + sum(  Vm[b,co]*Vm[busIdx[lines[l].to],co]  *(-YftI[l]*cos(Va[b,co]-Va[busIdx[lines[l].to],  co])
              + YftR[l]*sin(Va[b,co]-Va[busIdx[lines[l].to],co]  )) for l in FromLines[b] )
        + sum( Vm[b,co]*Vm[busIdx[lines[l].from],co] *(-YtfI[l]*cos(Va[b,co]-Va[busIdx[lines[l].from],co])
              + YtfR[l]*sin(Va[b,co]-Va[busIdx[lines[l].from],co])) for l in ToLines[b]   )
        -(sum(baseMVA*Qgc[g] for g in BusGeners[b]) - buses[b].Qd) / baseMVA      #Sbus part
        ==0)
    end # of for: power flow balance loop

    # branch/lines flow limits

    nlinelim=0
    for l in 1:nline
      if lines[l].rateA!=0 && lines[l].rateA<1.0e10
        nlinelim += 1
        flowmax=(lines[l].rateA/baseMVA)^2

        #branch apparent power limits (from bus)
        Yff_abs2=YffR[l]^2+YffI[l]^2; Yft_abs2=YftR[l]^2+YftI[l]^2
        Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
        @NLconstraint( opfmodel,
	  Vm[busIdx[lines[l].from],co]^2 *
	  ( Yff_abs2*Vm[busIdx[lines[l].from],co]^2 + Yft_abs2*Vm[busIdx[lines[l].to],co]^2
	  + 2*Vm[busIdx[lines[l].from],co]*Vm[busIdx[lines[l].to],co]
               *(Yre*cos(Va[busIdx[lines[l].from],co]-Va[busIdx[lines[l].to],co])-Yim*sin(Va[busIdx[lines[l].from],co]-Va[busIdx[lines[l].to],co]))
	  )
          - flowmax <=0)

        #branch apparent power limits (to bus)
        Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
        Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
        @NLconstraint(
          opfmodel,
    	  Vm[busIdx[lines[l].to],co]^2 *
          ( Ytf_abs2*Vm[busIdx[lines[l].from],co]^2 + Ytt_abs2*Vm[busIdx[lines[l].to],co]^2
          + 2*Vm[busIdx[lines[l].from],co]*Vm[busIdx[lines[l].to],co]
               *(Yre*cos(Va[busIdx[lines[l].from],co]-Va[busIdx[lines[l].to],co])-Yim*sin(Va[busIdx[lines[l].from],co]-Va[busIdx[lines[l].to],co]))
          )
          - flowmax <=0)
      end
    end # of for: branch power limits


    @printf("Contingency %d -> Buses: %d  Lines: %d  Generators: %d\n", co, nbus, nline, ngen)
    println("     lines with limits:  ", nlinelim)

  end # of for: contingencies


  # minimize active power
#  @NLobjective(opfmodel,
#		  Min,
#		  sum( generators[i].coeff[generators[i].n] +
#		       sum(generators[i].coeff[generators[i].n-k]*(baseMVA*Pg[i])^k for k=1:generators[i].n-1)
#		       for i=1:ngen)
#		 )



  return opfmodel
end

  #######################################################

  #values = zeros(2*nbus+2*ngen)
  ## values[1:2*nbus+2*ngen] = readdlm("/sandbox/petra/work/installs/matpower5.1/vars2.txt")
  #values[1:2*nbus+2*ngen] = readdlm("/sandbox/petra/work/installs/matpower5.1/vars3_9241.txt")
  #d = JuMP.NLPEvaluator(opfmodel)
  #MathProgBase.initialize(d, [:Jac])

  #g = zeros(2*nbus+2*nlinelim)
  #MathProgBase.eval_g(d, g, values)
  #println("f=", MathProgBase.eval_f(d,values))

  #gmat=zeros(2*nbus+2*nlinelim)
  #gmat[1:end] = readdlm("/sandbox/petra/work/installs/matpower5.1/cons3_9241.txt")
  #println("diff: ", norm(gmat-g))

  #println(opfmodel)

  #############################################################

function scopf_outputAll(opfmodel, scopf_data)
  #shortcuts for compactness
  sd=scopf_data; opf_data=sd.opfdata
  lines = opf_data.lines; buses = opf_data.buses; generators = opf_data.generators; baseMVA = opf_data.baseMVA
  busIdx = opf_data.BusIdx; FromLines = opf_data.FromLines; ToLines = opf_data.ToLines; BusGeners = opf_data.BusGenerators;

  nbus  = length(buses); nline = length(lines); ngen  = length(generators)

  # OUTPUTING
  println("Objective value: ", objective_value(opfmodel), "USD/hr")
  VM=value.(getindex(opfmodel,:Vm)); VA=value.(getindex(opfmodel,:Va));
  PG=value.(getindex(opfmodel,:Pg)); QG=value.(getindex(opfmodel,:Qg));

  VM=VM[:,0]; VA=VA[:,0]; #base case

  EX=value.(getindex(opfmodel,:extra));
  EX=EX[:,0];

  # printing the first stage variables
  println("============================= BUSES ==================================")
  println("  Generator  |  extra ")   # |    P (MW)     Q (MVAr)")  #|         (load)   ")
  println("----------------------------------------------------------------------")
  for i in 1:ngen
      @printf("  %10d | %6.2f \n",generators[i].bus, EX[i])
  end
  println("\n")

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
  return

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
          #@printf("%7.4f   %7.4f    %7.4f \n", consRhs[idx], consRhs[idx]+flowmax,  flowmax)
        end
        nlim += 1
      end
    end
  end

  return
end

function main(args)
  if length(args) < 1
    println("Usage: julia scopf.jl case_name")
    println("Cases are in 'data' directory: case9 case30 case57 case118 case300 case1354pegase case2383wp case2736sp case2737sop case2746wop case2869pegase case3012wp  case3120sp case3375wp case9241pegase")
    return
  end

  max_iter = 100
  opfdata = acopf.opf_loaddata(args)

  scopfmodel = auglag_model(opfdata; rho=10)
  opfmodel, status = solve(scopfmodel)

  d = NLPEvaluator(scopfmodel)
  MOI.initialize(d, [:Grad, :Jac, :Hess])
  op_model = backend(scopfmodel).optimizer.model
  inner = backend(scopfmodel).optimizer.model.inner
  nvar = length(inner.x)

  IJ = MOI.hessian_lagrangian_structure(d)
  Hv_tmp = zeros(length(IJ))
  MOI.eval_hessian_lagrangian(d, Hv_tmp, inner.x, 1.0, inner.mult_g)

  # Make it upper-triangular.
  I = zeros(length(IJ))
  J = zeros(length(IJ))
  for i=1:length(IJ)
    I[i] = IJ[i][1]
    J[i] = IJ[i][2]
    if I[i] > J[i]
      I[i], J[i] = J[i], I[i]
    end
  end

  # Merge duplicates.
  Ih, Jh, Vh = findnz(sparse(I, J, [Int[i] for i=1:length(I)], nvar, nvar, vcat))
  Hv = [sum(Hv_tmp[Vh[i]]) for i=1:length(Ih)]

  return scopfmodel, Hv

  # if status == MOI.LOCALLY_SOLVED
  #   scopf_outputAll(opfmodel,scopfdata)
  # end
end


