module acopf
include("opfdata.jl")
include("jumpmodel.jl")
include("IpoptTest.jl")
using CuArrays, CUDAnative
using ForwardDiff
using TimerOutputs

export create_arrays, objective, constraints 
export myseed!

# GPU and CPU kernels
include("kernels.jl")

mutable struct spmat{T}
  colptr::CuVector{Int64}
  rowval::CuVector{Int64}
  nzval::CuVector{T}

  function spmat{T}(colptr::Vector{Int64}, rowval::Vector{Int64}, nzval::Vector{T}) where T
    return new(CuVector{Int64}(colptr), CuVector{Int64}(rowval), CuVector{T}(nzval))
  end
end

mutable struct CompArrays
  cuPg ; cuQg ; cuVa ; cuVm
  baseMVA ;nbus ; nline ; ngen
  coeff0 ; coeff1 ; coeff2
  viewToR ; viewFromR ; viewToI ; viewFromI
  viewVmTo ; viewVmFrom ; viewVaTo ; viewVaFrom
  viewYffR ; viewYttR ; viewYffI ; viewYttI
  cuYffR ; cuYffI ; cuYttR ; cuYttI ; cuYftR ; cuYftI ; cuYtfR ; cuYtfI ; cuYshR ; cuYshI
  viewPg ; viewQg ; sumPg; sumQg; sizeBusGeners
  cuPd ; cuQd
  cuflowmax
  Yff_abs2 ; Yft_abs2 ; Ytf_abs2 ; Ytt_abs2
  Yrefrom ; Yimfrom ; Yreto ; Yimto
  viewVmToFromLines ; viewcuYftRFromLines ; viewVaToFromLines ; viewcuYftIFromLines ; viewVmFromToLines ; viewcuYtfRToLines ; viewVaFromToLines ; viewcuYtfIToLines
  sizeFromLines ; sizeToLines
  cuBusGeners; mapfromlines; maptolines
  stype
end

function myseed!(duals::AbstractArray{ForwardDiff.Dual{T,V,N}}, x,
               seeds::AbstractArray{ForwardDiff.Partials{N,V}}, timeroutput) where {T,V,N}

  @timeit timeroutput "myseed 1" begin
    for i in 1:size(duals,1)
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
    end
  end
  # @timeit timeroutput "myseed 2" begin
  #   zeroseed::ForwardDiff.Partials{N,V} = zero(ForwardDiff.Partials{N,V}) 
  #   for i in N+1:size(duals,1)
  #       duals[i] = ForwardDiff.Dual{T,V,N}(x[i], zeroseed)
  #   end
  # end
    return duals
end

function create_arrays(cuPg::T, cuQg::T, cuVa::T, cuVm::T, opf_data, timeroutput, stype = CuArray) where T
  @timeit timeroutput "create arrays" begin
  #shortcuts for compactness
  lines = opf_data.lines; buses = opf_data.buses; generators = opf_data.generators; baseMVA = opf_data.baseMVA
  busIdx = opf_data.BusIdx; FromLines = opf_data.FromLines; ToLines = opf_data.ToLines; BusGeners = opf_data.BusGenerators;
  nbus  = length(buses); nline = length(lines); ngen  = length(generators)

  arraytype = stype
  if arraytype == CuArray
    TVector = CuVector
  elseif arraytype == Array
    TVector = Vector
  else
    error("Unkown array type $arraytype.")
  end

  ## Arrays for objective

  # Penalty coefficients
  coeff0 = TVector{Float64}(undef, ngen)
  for (i,v) in enumerate(generators)
    coeff0[i] = v.coeff[v.n]
  end
  coeff1 = TVector{Float64}(undef, ngen)
  for (i,v) in enumerate(generators)
    coeff1[i] = v.coeff[v.n - 1]
  end
  coeff2 = TVector{Float64}(undef, ngen)
  for (i,v) in enumerate(generators)
    coeff2[i] = v.coeff[v.n - 2]
  end
  
  ## Arrays for constraints

  # Demand arrays
  cuPd = TVector{Float64}(undef, nbus)
  for (b,v) in enumerate(cuPd)
    cuPd[b] = buses[b].Pd
  end
  cuQd = TVector{Float64}(undef, nbus)
  for (b,v) in enumerate(cuQd)
    cuQd[b] = buses[b].Qd
  end

  # Indices
  cuBusGeners = Array{TVector{Int64}}(undef, nbus)
  for (b,v) in enumerate(BusGeners) cuBusGeners[b] = cu(BusGeners[b]) end
  cuFromLines = Array{TVector{Int64}}(undef, nbus)
  for (b,v) in enumerate(FromLines) cuFromLines[b] = cu(FromLines[b]) end
  cuToLines = Array{TVector{Int64}}(undef, nbus)
  for (b,v) in enumerate(ToLines) cuToLines[b] = cu(ToLines[b]) end
  
  sizeFromLines = CuArray{Int64,1,Nothing}(size.(cuFromLines,1))
  sizeToLines = CuArray{Int64,1,Nothing}(size.(cuToLines,1))

  # Views
  connect = maximum(size.(cuBusGeners,1))
  sumQg  = T(undef, nbus)
  sumPg  = T(undef, nbus)
  
  function spfill(from::T, ranges, nbus) where T
    k = 1
    colptr = Vector{Int64}()
    rowval = Vector{Int64}()
    nzval  = Vector{T.parameters[1]}()
    for b in 1:nbus
      push!(colptr,k)
      for (j,i) in enumerate(ranges[b])
        # to[j,b] = from[i] 
        push!(nzval, from[i])
        push!(rowval, i)
        k += 1
      end
    end
    push!(colptr,k)
    return spmat{T.parameters[1]}(colptr, rowval, nzval)
  end
  viewPg = spfill(cuPg, cuBusGeners, nbus)
  viewQg = spfill(cuQg, cuBusGeners, nbus)
  sizeBusGeners = TVector{Int64}(size.(cuBusGeners,1))
  
  # Bus voltage

  mapbus2lineto = TVector{Int64}([busIdx[lines[l].to] for l in 1:nline]) 
  mapbus2linefrom = TVector{Int64}([busIdx[lines[l].from] for l in 1:nline])
  viewVmTo = view(cuVm, mapbus2lineto)
  viewVmFrom = view(cuVm, mapbus2linefrom)
  viewVaTo = view(cuVa, mapbus2lineto)
  viewVaFrom = view(cuVa, mapbus2linefrom)

  # Branch admitances
  YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)
  cuYffR = TVector{Float64}(YffR) ; cuYffI = TVector{Float64}(YffI) ; cuYttR = TVector{Float64}(YttR) ; cuYttI = TVector{Float64}(YttI) ; cuYftR = TVector{Float64}(YftR)
  cuYftI = TVector{Float64}(YftI) ; cuYtfR = TVector{Float64}(YtfR) ; cuYtfI = TVector{Float64}(YtfI) ; cuYshR = TVector{Float64}(YshR) ; cuYshI = TVector{Float64}(YshI)

  viewYffR = TVector{Float64}(undef, nbus) ; 
  for b in 1:nbus viewYffR[b] = sum(view(cuYffR, cuFromLines[b])) end
  viewYttR = TVector{Float64}(undef, nbus) ; 
  for b in 1:nbus viewYttR[b] = sum(view(cuYttR, cuToLines[b])) end
  viewYffI = TVector{Float64}(undef, nbus) ; 
  for b in 1:nbus viewYffI[b] = sum(.- view(cuYffI, cuFromLines[b])) end
  viewYttI = TVector{Float64}(undef, nbus) ; 
  for b in 1:nbus viewYttI[b] = sum(.- view(cuYttI, cuToLines[b])) end

  viewToR = T(undef, nbus)  
  viewFromR = T(undef, nbus)  
  viewToI = T(undef, nbus)
  viewFromI = T(undef, nbus)  

  # Line limits
  culinelimit = TVector{Float64}(undef, nline)
  nlinelimit = 0
  for l in 1:nline
    if lines[l].rateA!=0 && lines[l].rateA<1.0e10
      nlinelimit += 1
      culinelimit[l] = 1.0
    else
      culinelimit[l] = 0.0
    end
  end
  curate = TVector{Float64}(undef, nline)
  for l in 1:nline curate[l] = lines[l].rateA end
  cuflowmax= TVector{Float64}(undef, nline)
  cuflowmax .= (curate ./ baseMVA).^2 

  Yff_abs2=TVector{Float64}(undef, nline); Yft_abs2=TVector{Float64}(undef, nline); 
  Ytf_abs2=TVector{Float64}(undef, nline); Ytt_abs2=TVector{Float64}(undef, nline); 
  Yrefrom=TVector{Float64}(undef, nline); Yimfrom=TVector{Float64}(undef, nline); 
  Yreto=TVector{Float64}(undef, nline); Yimto=TVector{Float64}(undef, nline); 
  Yff_abs2 .= cuYffR.^2 .+ cuYffI.^2; Yft_abs2 .= cuYftR.^2 .+ cuYftI.^2
  Ytf_abs2 .= cuYtfR.^2 .+ cuYtfI.^2; Ytt_abs2 .= cuYttR.^2 .+ cuYttI.^2
  Yrefrom .= cuYffR .* cuYftR .+ cuYffI .* cuYftI; Yimfrom .= .- cuYffR .* cuYftI .+ cuYffI .* cuYftR
  Yreto   .= cuYtfR .* cuYttR .+ cuYtfI .* cuYttI; Yimto   .= .- cuYtfR .* cuYttI .+ cuYtfI .* cuYttR

  # Voltage and bus connectivity

  mapfromlines = Array{TVector{Int64},1}(undef, nbus)
  
  for b in 1:nbus mapfromlines[b] = mapbus2lineto[cuFromLines[b]] end
  viewVmToFromLines   = spfill(cuVm, mapfromlines, nbus)
  viewcuYftRFromLines = spfill(cuYftR, cuFromLines, nbus)
  viewVaToFromLines   = spfill(cuVa, mapfromlines, nbus)
  viewcuYftIFromLines = spfill(cuYftI, cuFromLines, nbus)

  maptolines = Array{TVector{Int64},1}(undef, nbus)

  for b in 1:nbus maptolines[b] = mapbus2linefrom[cuToLines[b]] end
  viewVmFromToLines = spfill(cuVm, maptolines, nbus)
  viewcuYtfRToLines = spfill(cuYtfR, cuToLines, nbus)
  viewVaFromToLines = spfill(cuVa, maptolines, nbus) 
  viewcuYtfIToLines = spfill(cuYtfI, cuToLines, nbus)

  ret = CompArrays(cuPg, cuQg, cuVa, cuVm, baseMVA, nbus, nline, ngen, 
                    coeff0, coeff1, coeff2, 
                    viewToR, viewFromR, viewToI, viewFromI,
                    viewVmTo, viewVmFrom, viewVaTo, viewVaFrom,
                    viewYffR, viewYttR, viewYffI, viewYttI,
                    cuYffR, cuYffI, cuYttR, cuYttI, cuYftR, cuYftI, cuYtfR, cuYtfI, cuYshR, cuYshI,
                    viewPg, viewQg, sumPg, sumQg, sizeBusGeners,
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
                    viewVmFromToLines, viewcuYtfRToLines, viewVaFromToLines, viewcuYtfIToLines,
                    sizeFromLines, sizeToLines,
                    cuBusGeners, mapfromlines, maptolines,
                    stype)
  end
  return ret
end

function update_arrays!(arrays::CompArrays, cuPg::T, cuQg::T, cuVa::T, cuVm::T, timeroutput) where T
  @timeit timeroutput "update arrays" begin
  arrays.cuPg .= cuPg
  arrays.cuQg .= cuQg
  arrays.cuVa .= cuVa
  arrays.cuVm .= cuVm

  function spfill!(to, from::T, ranges, nbus) where T
    k = 1
    for b in 1:nbus
      for (j,i) in enumerate(ranges[b])
        to.nzval[k] = from[i]
        k += 1
      end
    end
    return nothing
  end
  spfill!(arrays.viewPg, cuPg, arrays.cuBusGeners, arrays.nbus)
  spfill!(arrays.viewQg, cuQg, arrays.cuBusGeners, arrays.nbus)

  spfill!(arrays.viewVmToFromLines, cuVm, arrays.mapfromlines, arrays.nbus)
  spfill!(arrays.viewVaToFromLines, cuVa, arrays.mapfromlines, arrays.nbus)

  spfill!(arrays.viewVmFromToLines, cuVm, arrays.maptolines, arrays.nbus)
  spfill!(arrays.viewVaFromToLines, cuVa, arrays.maptolines, arrays.nbus) 
  end
end

function objective(arrays::CompArrays, timeroutput) where T
  @timeit timeroutput "objective" begin
  # minimize active power
  return sum(arrays.coeff2 .* (arrays.baseMVA .* arrays.cuPg).^2 
          .+ arrays.coeff1 .* (arrays.baseMVA .* arrays.cuPg)
          .+ arrays.coeff0)
  end
end

function constraints(rbalconst::T, ibalconst::T, limitsto::T, limitsfrom::T, arrays::CompArrays, timeroutput::TimerOutput) where T
  @timeit timeroutput "constraints" begin
  
  @timeit timeroutput "cuda kernels" begin
  nbus = arrays.nbus

  arrays.sumPg .= 0.0
  arrays.sumQg .= 0.0 
  arrays.viewToR   .= 0.0
  arrays.viewFromR .= 0.0
  arrays.viewToI   .= 0.0
  arrays.viewFromI .= 0.0
  arraytype = arrays.stype

  # nthreads=nbus
  # nblocks=1
  nthreads=256
  # println("Threads: ", nthreads)
  nblocks=ceil(Int64, nbus/nthreads)
  # println("Blocks: ", nblocks)

  kernels.@sync arraytype begin
  kernels.@dispatch arraytype threads=nthreads blocks=nblocks sumPg(arrays.sumPg, arrays.viewPg.colptr, arrays.viewPg.nzval)
  kernels.@dispatch arraytype threads=nthreads blocks=nblocks sumQg(arrays.sumQg, arrays.viewQg.colptr, arrays.viewQg.nzval)

  kernels.@dispatch arraytype threads=nthreads blocks=nblocks term1(arrays.viewToR, arrays.cuVm, 
                                        arrays.viewVmToFromLines.colptr, arrays.viewVmToFromLines.nzval,
                                        arrays.viewcuYftRFromLines.colptr, arrays.viewcuYftRFromLines.nzval,
                                        arrays.cuVa, 
                                        arrays.viewVaToFromLines.colptr, arrays.viewVaToFromLines.nzval, 
                                        arrays.viewcuYftIFromLines.colptr, arrays.viewcuYftIFromLines.nzval,
                                        arrays.sizeFromLines) 

  kernels.@dispatch arraytype threads=nthreads blocks=nblocks term2(arrays.viewFromR, arrays.cuVm, 
                                        arrays.viewVmFromToLines.colptr, arrays.viewVmFromToLines.nzval,
                                        arrays.viewcuYtfRToLines.colptr, arrays.viewcuYtfRToLines.nzval,
                                        arrays.cuVa, 
                                        arrays.viewVaFromToLines.colptr, arrays.viewVaFromToLines.nzval,  
                                        arrays.viewcuYtfIToLines.colptr, arrays.viewcuYtfIToLines.nzval,
                                        arrays.sizeToLines) 

  kernels.@dispatch arraytype threads=nthreads blocks=nblocks term3(arrays.viewToI, arrays.cuVm, 
                                        arrays.viewVmToFromLines.colptr, arrays.viewVmToFromLines.nzval,
                                        arrays.viewcuYftIFromLines.colptr, arrays.viewcuYftIFromLines.nzval,
                                        arrays.cuVa, 
                                        arrays.viewVaToFromLines.colptr, arrays.viewVaToFromLines.nzval, 
                                        arrays.viewcuYftRFromLines.colptr, arrays.viewcuYftRFromLines.nzval,
                                        arrays.sizeFromLines) 

  kernels.@dispatch arraytype threads=nthreads blocks=nblocks term4(arrays.viewFromI, arrays.cuVm, 
                                        arrays.viewVmFromToLines.colptr, arrays.viewVmFromToLines.nzval,
                                        arrays.viewcuYtfIToLines.colptr, arrays.viewcuYtfIToLines.nzval,
                                        arrays.cuVa, 
                                        arrays.viewVaFromToLines.colptr, arrays.viewVaFromToLines.nzval, 
                                        arrays.viewcuYtfRToLines.colptr, arrays.viewcuYtfRToLines.nzval,
                                        arrays.sizeToLines) 
  end
  end
  @timeit timeroutput "balance constraints" begin
  rbalconst .= (
               (arrays.viewYffR .+ arrays.viewYttR .+ arrays.cuYshR) .* arrays.cuVm.^2 
               .+ arrays.viewToR    # kernel term 1 
               .+ arrays.viewFromR  # kernel term 2
               .- (((arrays.sumPg .* arrays.baseMVA) .- arrays.cuPd) ./ arrays.baseMVA) 
               )

  ibalconst .= (
               (arrays.viewYffI .+ arrays.viewYttI .- arrays.cuYshI) .* arrays.cuVm.^2 
               .+ arrays.viewToI    # kernel term 3
               .+ arrays.viewFromI  # kernel term 4
               .- (((arrays.sumQg .* arrays.baseMVA) .- arrays.cuQd) ./ arrays.baseMVA) 
               )
  end

  if arraytype == CuArray
    mycos = CUDAnative.cos
    mysin = CUDAnative.sin
  elseif arraytype == Array
    mycos = cos
    mysin = sin
  else
    error("Unkown array type $arraytype.")
  end

  # branch apparent power limits (from bus)
  @timeit timeroutput "line constraints" begin
  limitsto .= (
              (arrays.viewVmFrom.^2 
              .* (arrays.Yff_abs2 .* arrays.viewVmFrom.^2 .+ arrays.Yft_abs2 .* arrays.viewVmTo.^2
              .+ 2.0 .* arrays.viewVmFrom .* arrays.viewVmTo 
              .* (arrays.Yrefrom .* mycos.(arrays.viewVaFrom .- arrays.viewVaTo) 
                  .- arrays.Yimfrom .* mysin.(arrays.viewVaFrom .- arrays.viewVaTo)))
              .- arrays.cuflowmax)
              )
  # branch apparent power limits (to bus)
  limitsfrom .= ( 
                (arrays.viewVmTo.^2 
                .* (arrays.Ytf_abs2 .* arrays.viewVmFrom.^2 .+ arrays.Ytt_abs2 .* arrays.viewVmTo.^2
                .+ 2.0 .* arrays.viewVmFrom .* arrays.viewVmTo 
                .* (arrays.Yreto .* mycos.(arrays.viewVaFrom - arrays.viewVaTo) 
                    .- arrays.Yimto .* mysin.(arrays.viewVaFrom .- arrays.viewVaTo)))
                .- arrays.cuflowmax)
                )
  end
  end
  return nothing
end

function benchmark(opfdata, Pg, Qg, Vm, Va, npartials, mpartials, loops, timeroutput, stype)
  
  t1s{N} =  ForwardDiff.Dual{Nothing,Float64, N} where N
  t2s{M,N} =  ForwardDiff.Dual{Nothing,t1s{N}, M} where {N, M}

  arraytype = stype
  if arraytype == CuArray
    T = arraytype{Float64, 1, Nothing}
    t1sT = arraytype{t1s{npartials}, 1, Nothing}
    t2sT = arraytype{t2s{mpartials, npartials}, 1, Nothing}
  end
  if arraytype == Array
    T = arraytype{Float64, 1}
    t1sT = arraytype{t1s{npartials}, 1}
    t2sT = arraytype{t2s{mpartials, npartials}, 1}
  end

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
  t1scuPg = acopf.myseed!(t1sT(undef, size(Pg,1)), Pg, t1sseeds, timeroutput)
  t1scuQg = ForwardDiff.seed!(t1sT(undef, size(Qg,1)), Qg)
  t1scuVa = ForwardDiff.seed!(t1sT(undef, size(Va,1)), Va)
  t1scuVm = ForwardDiff.seed!(t1sT(undef, size(Vm,1)), Vm)
  end


  # # t2sseedvec = Array{t1s{npartials},1}(undef, mpartials)
  # t2sseedvec = Array{t1s{npartials},1}(undef, npartials)
  # # t2sseeds = Array{ForwardDiff.Partials{mpartials,t1s{npartials}},1}(undef, size(Pg,1))
  # t2sseeds = Array{ForwardDiff.Partials{mpartials,t1s{npartials}},1}(undef, mpartials)
  # t2sseedvec .= 0.0
  # for i in 1:mpartials
  #   t2sseedvec[i] = 1.0
  #   # t2sseeds[i] = ForwardDiff.Partials{mpartials, t1s{npartials}}(NTuple{mpartials, t1s{npartials}}(t2sseedvec))
  #   t2sseeds[i] = ForwardDiff.Partials{mpartials, t1s{npartials}}(NTuple{npartials, t1s{npartials}}(t2sseedvec))
  #   t2sseedvec[i] = 0.0
  # end
  # for i in mpartials+1:size(Pg,1)
  #   t2sseeds[i] = ForwardDiff.Partials{mpartials, t1s{npartials}}(NTuple{mpartials, t1s{npartials}}(t2sseedvec))
  # end
  # t2sseedtup = NTuple{size(Pg,1), ForwardDiff.Partials{size(Pg,1), Float64}}(t1sseeds)
  @timeit timeroutput "t2s seeding" begin
  # t2scuPg = acopf.myseed!(t2sT(undef, size(Pg,1)), t1scuPg, t2sseeds, timeroutput)
  # t2scuPg = acopf.myseed!(t2sT(undef, size(Pg,1)), t1scuPg, t2sseeds, timeroutput)
  # t2scuPg = ForwardDiff.seed!(t2sT(undef, size(Qg,1)), t1scuPg)
  # t2scuPg = ForwardDiff.seed!(t2sT(undef, size(Pg,1)), t1scuPg)
  # t2scuQg = ForwardDiff.seed!(t2sT(undef, size(Qg,1)), t1scuQg)
  # t2scuVa = ForwardDiff.seed!(t2sT(undef, size(Va,1)), t1scuVa)
  # t2scuVm = ForwardDiff.seed!(t2sT(undef, size(Vm,1)), t1scuVm)
  t2scuPg = t2sT(undef, size(Pg,1))
  t2scuQg = t2sT(undef, size(Qg,1))
  t2scuVa = t2sT(undef, size(Va,1))
  t2scuVm = t2sT(undef, size(Vm,1))
  end
  # @show t2scuPg

  # Pg0,Qg0,Vm0,Va0 = acopf.acopf_initialPt_IPOPT(opfdata)
  # cuPg = ForwardDiff.seed!(arraytype{ForwardDiff.Dual{Nothing,Float64,size(Pg,1)}, 1, Nothing}(undef, size(Pg,1)), Pg0, seedtup)
  # cuQg = ForwardDiff.seed!(arraytype{ForwardDiff.Dual{Nothing,Float64,size(Pg,1)}, 1, Nothing}(undef, size(Qg,1)), Qg0)
  # cuVa = ForwardDiff.seed!(arraytype{ForwardDiff.Dual{Nothing,Float64,size(Pg,1)}, 1, Nothing}(undef, size(Va,1)), Va0)
  # cuVm = ForwardDiff.seed!(arraytype{ForwardDiff.Dual{Nothing,Float64,size(Pg,1)}, 1, Nothing}(undef, size(Vm,1)), Vm0)
  T = typeof(t1scuQg)
  t1srbalconst = T(undef, length(opfdata.buses))
  t1sibalconst = T(undef, length(opfdata.buses))
  t1slimitsto = T(undef, length(opfdata.lines))
  t1slimitsfrom = T(undef, length(opfdata.lines))
  println("Create t1s arrays")
  @timeit timeroutput "Create t1s arrays" begin
  t1sarrays = acopf.create_arrays(t1scuPg, t1scuQg, t1scuVa, t1scuVm, opfdata, timeroutput, stype)
  end
  println("Initial t1s objective")
  @timeit timeroutput "Initial t1s objective" begin
  t1sPg = acopf.objective(t1sarrays, timeroutput)
  end
  println("t1s objective")
  @timeit timeroutput "t1s objective" begin
    for i in 1:loops
      t1sPg = acopf.objective(t1sarrays, timeroutput)
    end
  end
  println("Initial t1s constraints")
  @timeit timeroutput "Initial t1s constraints" begin
  acopf.constraints(t1srbalconst, t1sibalconst, t1slimitsto, t1slimitsfrom, t1sarrays, timeroutput)
  end
  println("t1s constraints $npartials")
  @timeit timeroutput "t1s constraints $npartials" begin
    for i in 1:loops
      acopf.constraints(t1srbalconst, t1sibalconst, t1slimitsto, t1slimitsfrom, t1sarrays, timeroutput)
    end
  end
  T = typeof(t2scuQg)
  t2srbalconst = T(undef, length(opfdata.buses))
  t2sibalconst = T(undef, length(opfdata.buses))
  t2slimitsto = T(undef, length(opfdata.lines))
  t2slimitsfrom = T(undef, length(opfdata.lines))
  println("Create t2s arrays")
  @timeit timeroutput "Create t2s arrays" begin
  t2sarrays = acopf.create_arrays(t2scuPg, t2scuQg, t2scuVa, t2scuVm, opfdata, timeroutput, arraytype)
  end
  println("Initial t2s objective")
  @timeit timeroutput "Initial t2s objective" begin
  t2sPg = acopf.objective(t2sarrays, timeroutput);
  end
  println("t2s objective")
  @timeit timeroutput "t2s objective" begin
    for i in 1:loops
      t2sPg = acopf.objective(t2sarrays, timeroutput)
    end
  end
  println("Initial t2s constraints")
  @timeit timeroutput "Initial t2s constraints" begin
  acopf.constraints(t2srbalconst, t2sibalconst, t2slimitsto, t2slimitsfrom, t2sarrays, timeroutput)
  end
  println("t2s constraints")
  @timeit timeroutput "t2s constraints $npartials" begin
    for i in 1:loops
      acopf.constraints(t2srbalconst, t2sibalconst, t2slimitsto, t2slimitsfrom, t2sarrays, timeroutput)
    end
  end
  return nothing
end
end

