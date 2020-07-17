using JuMP
using Ipopt
using Hiop
using Printf
using DelimitedFiles
using acopf

struct SCOPFData
  opfdata
  lines_off::Array
  #Float64::gener_ramp #generator ramp limit for contingency (percentage)
end

function solve(opfmodel)
  # 
  # Initial point - needed especially for pegase cases
  #
  # Pg0,Qg0,Vm0,Va0 = acopf.initialPt_IPOPT(scopf_data.opfdata)
  # setvalue(getindex(opfmodel, :Pg), Pg0)  
  # setvalue(getindex(opfmodel, :Qg), Qg0)
  # extra_jump=getindex(opfmodel, :extra)
  # Vm_jump=getindex(opfmodel, :Vm)
  # Va_jump=getindex(opfmodel, :Va)

  # setvalue(Vm_jump[:,0], Vm0)    
  # setvalue(Va_jump[:,0], Va0)    
  # setvalue(extra_jump[:,0], 0.025*Pg0)   
  # ncont=length(scopf_data.lines_off); nbus=length(scopf_data.opfdata.buses)

  # #println(opfmodel)

  # for c in 1:ncont
  #   opfm1=opf_loaddata(ARGS[1], scopf_data.lines_off[c])
  #   Pg0,Qg0,Vm0,Va0 = acopf_initialPt_IPOPT(opfm1)
  #   setvalue(extra_jump[:,c], 0.025*Pg0)   
  #   setvalue(Vm_jump[:,c], Vm0)    
  #   setvalue(Va_jump[:,c], Va0)    
  # end


  # status = solve(opfmodel)

  optimize!(opfmodel)
  status = termination_status(opfmodel)
  if status != MOI.LOCALLY_SOLVED
      println("Could not solve the model to optimality.")
  end


  return opfmodel, status
  #scopf_outputAll(opfmodel, scopf_data)
end

function model(scopf_data; max_iter=100)
  sd=scopf_data
  Pg0,Qg0,Vm0,Va0 = acopf.initialPt_IPOPT(sd.opfdata)
  lines = sd.opfdata.lines; buses = sd.opfdata.buses; generators = sd.opfdata.generators; baseMVA = sd.opfdata.baseMVA
  busIdx = sd.opfdata.BusIdx; FromLines = sd.opfdata.FromLines; ToLines = sd.opfdata.ToLines; BusGeners = sd.opfdata.BusGenerators;
  ncont=length(sd.lines_off); nbus=length(sd.opfdata.buses); ngen=length(sd.opfdata.generators)

  generators=sd.opfdata.generators; buses=sd.opfdata.buses; baseMVA = sd.opfdata.baseMVA

  opfmodel = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => max_iter))

  println("Considering ", ncont, " contingengies")

  @variable(opfmodel, generators[i].Pmin <= Pg[i=1:ngen] <= generators[i].Pmax, start = Pg0[i])
  @variable(opfmodel, 0<=extra[i=1:ngen,0:ncont]<=0.00*generators[i].Pmax, start = 0.0)
  @variable(opfmodel, generators[i].Qmin <= Qg[i=1:ngen] <= generators[i].Qmax, start = Qg0[i])

  @NLobjective(opfmodel, Min, sum( generators[i].coeff[generators[i].n-2]*(baseMVA*(Pg[i]+extra[i,c]))^2 
			             +generators[i].coeff[generators[i].n-1]*(baseMVA*(Pg[i]+extra[i,c]))
				     +generators[i].coeff[generators[i].n  ] for i=1:ngen, c=0:ncont) / (1+ncont))


  @variable(opfmodel, buses[i].Vmin <= Vm[i=1:nbus, c=0:ncont] <= buses[i].Vmax, start = Vm0[i])
  @variable(opfmodel, Va[i=1:nbus, c=0:ncont], start = Va0[i])
  
  #fix the voltage angle at the reference bus
  for c in 0:ncont
    set_lower_bound(Va[sd.opfdata.bus_ref,c], buses[sd.opfdata.bus_ref].Va)
    set_upper_bound(Va[sd.opfdata.bus_ref,c], buses[sd.opfdata.bus_ref].Va)
  end

  @constraint(opfmodel, ex[i=1:ngen,co=0:ncont], generators[i].Pmin <= Pg[i] + extra[i,co] <= generators[i].Pmax)
  for co=0:ncont
    #branch admitances
    # if co==0
      YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(sd.opfdata.lines, sd.opfdata.buses, sd.opfdata.baseMVA)
      lines=sd.opfdata.lines
      busIdx=sd.opfdata.BusIdx;FromLines=sd.opfdata.FromLines; ToLines=sd.opfdata.ToLines; BusGeners=sd.opfdata.BusGenerators
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
        -(sum(baseMVA*(Pg[g]+extra[g,co]) for g in BusGeners[b]) - buses[b].Pd) / baseMVA      # Sbus part
        ==0)

      #imaginary part
      @NLconstraint( opfmodel,
        ( sum(-YffI[l] for l in FromLines[b]) + sum(-YttI[l] for l in ToLines[b]) - YshI[b] ) * Vm[b,co]^2 
        + sum(  Vm[b,co]*Vm[busIdx[lines[l].to],co]  *(-YftI[l]*cos(Va[b,co]-Va[busIdx[lines[l].to],  co]) 
              + YftR[l]*sin(Va[b,co]-Va[busIdx[lines[l].to],co]  )) for l in FromLines[b] )
        + sum( Vm[b,co]*Vm[busIdx[lines[l].from],co] *(-YtfI[l]*cos(Va[b,co]-Va[busIdx[lines[l].from],co]) 
              + YtfR[l]*sin(Va[b,co]-Va[busIdx[lines[l].from],co])) for l in ToLines[b]   )
        -(sum(baseMVA*Qg[g] for g in BusGeners[b]) - buses[b].Qd) / baseMVA      #Sbus part
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

function main()
  if length(ARGS) < 1
    println("Usage: julia scopf_main.jl case_name lines_indexes")
    println("Cases are in 'data' directory: case9 case30 case57 case118 case300 case1354pegase case2383wp case2736sp case2737sop case2746wop case2869pegase case3012wp  case3120sp case3375wp case9241pegase")
    return
  end
  
  max_iter = 100
  opfdata = acopf.opf_loaddata(ARGS[1])

  lines_off=Array{acopf.Line}(undef, length(ARGS)-1)
  for l in 1:length(lines_off)
    lines_off[l] = opfdata.lines[parse(Int,ARGS[l+1])]
  end
  scopfdata = SCOPFData(opfdata, lines_off)
  @assert length(lines_off) == length(ARGS)-1
  scopfmodel = model(scopfdata)
  opfmodel, status = solve(scopfmodel)
  if status == MOI.LOCALLY_SOLVED
    scopf_outputAll(opfmodel,scopfdata)
  end
end

main()

