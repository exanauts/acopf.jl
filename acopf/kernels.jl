module kernels
  using CUDAnative, CuArrays
  export sync, dispatch
  macro sync(type, expr)
    ex = quote 
      if $type == CuArray
        CuArrays.@sync begin
        $expr
        end
      end
      if $type == Array
        $expr
      end
    end
    return esc(ex)
  end

  macro dispatch(type, threads, blocks, expr)
    cuda = Meta.parse("kernels.cuda_$expr")
    cpu = Meta.parse("kernels.cpu_$expr")
    ex = nothing
    ex = quote 
      if $type == CuArray
          @cuda $threads $blocks $cuda
      end
      if $type == Array
          $cpu
      end
    end
    return esc(ex)
  end
  function cuda_term1(viewToR, cuVm, colptrVm, nzvalVm, colptrYftR, nzvalYftR,
                              cuVa, colptrVa, nzvalVa, colptrYftI, nzvalYftI,
                              sizeFromLines) 
      index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      stride = blockDim().x * gridDim().x
      for b in index:stride:size(viewToR,1)
        for (i,c) in enumerate(colptrVm[b]:colptrVm[b+1]-1)
          @inbounds viewToR[b] += cuVm[b] * nzvalVm[c] * 
                                  (  nzvalYftR[c] * CUDAnative.cos(cuVa[b] - nzvalVa[c]) 
                                   + nzvalYftI[c] * CUDAnative.sin(cuVa[b] - nzvalVa[c])
                                  ) 
        end
      end
      return nothing
  end

  function cuda_term2(viewFromR, cuVm, colptrVm, nzvalVm, colptrYtfR, nzvalYtfR,
                                  cuVa, colptrVa, nzvalVa, colptrYtfI, nzvalYtfI,
                                  sizeToLines) 
      index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      stride = blockDim().x * gridDim().x
      for b in index:stride:size(viewFromR,1)
        for (i,c) in enumerate(colptrVm[b]:colptrVm[b+1]-1)
          @inbounds viewFromR[b] += cuVm[b] * nzvalVm[c] * 
                                    (  nzvalYtfR[c] * CUDAnative.cos(cuVa[b] - nzvalVa[c]) 
                                     + nzvalYtfI[c] * CUDAnative.sin(cuVa[b] - nzvalVa[c])
                                    ) 
        end
      end
      return nothing
  end

  function cuda_term3(viewToI, cuVm, colptrVm, nzvalVm, colptrYftI, nzvalYftI,
                              cuVa, colptrVa, nzvalVa, colptrYftR, nzvalYftR,
                              sizeFromLines) 
      index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      stride = blockDim().x * gridDim().x
      for b in index:stride:size(viewToI,1)
        for (i,c) in enumerate(colptrVm[b]:colptrVm[b+1]-1)
          @inbounds viewToI[b] += cuVm[b] * nzvalVm[c] * 
                                  (  ( -nzvalYftI[c]) * CUDAnative.cos(cuVa[b] - nzvalVa[c]) 
                                      + nzvalYftR[c] * CUDAnative.sin(cuVa[b] - nzvalVa[c])
                                  ) 
        end
      end
      return nothing
  end

  function cuda_term4(viewFromI, cuVm, colptrVm, nzvalVm, colptrYtfI, nzvalYtfI,
                                  cuVa, colptrVa, nzvalVa, colptrYtfR, nzvalYtfR,
                                  sizeToLines) 
      index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      stride = blockDim().x * gridDim().x
      for b in index:stride:size(viewFromI,1)
        for (i,c) in enumerate(colptrVm[b]:colptrVm[b+1]-1)
          @inbounds viewFromI[b] += cuVm[b] * nzvalVm[c] * 
                                    (  ( -nzvalYtfI[c]) * CUDAnative.cos(cuVa[b] - nzvalVa[c]) 
                                        + nzvalYtfR[c] * CUDAnative.sin(cuVa[b] - nzvalVa[c])
                                    ) 
        end
      end
      return nothing
  end
  function cuda_sumPg(sumPg, colptr, nzval)
      index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      stride = blockDim().x * gridDim().x
      for b in index:stride:size(sumPg,1)
        for c in colptr[b]:colptr[b+1]-1
            @inbounds sumPg[b] += nzval[c]
        end
      end
      return nothing
  end
  function cuda_sumQg(sumQg, colptr, nzval)
      index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      stride = blockDim().x * gridDim().x
      for b in index:stride:size(sumQg,1)
        for c in colptr[b]:colptr[b+1]-1
            @inbounds sumQg[b] += nzval[c]
        end
      end
      return nothing
  end
  function cpu_term1(viewToR, cuVm, colptrVm, nzvalVm, colptrYftR, nzvalYftR,
                              cuVa, colptrVa, nzvalVa, colptrYftI, nzvalYftI,
                              sizeFromLines) 
      Threads.@threads for b in 1:size(viewToR,1)
        for (i,c) in enumerate(colptrVm[b]:colptrVm[b+1]-1)
          viewToR[b] += cuVm[b] * nzvalVm[c] * 
                                  (  nzvalYftR[c] * cos(cuVa[b] - nzvalVa[c])     
                                   + nzvalYftI[c] * sin(cuVa[b] - nzvalVa[c])
                                  ) 
        end
      end
      return nothing
  end

  function cpu_term2(viewFromR, cuVm, colptrVm, nzvalVm, colptrYtfR, nzvalYtfR,
                                  cuVa, colptrVa, nzvalVa, colptrYtfI, nzvalYtfI,
                                  sizeToLines) 
      for b in 1:size(viewFromR,1)
        for (i,c) in enumerate(colptrVm[b]:colptrVm[b+1]-1)
          viewFromR[b] += cuVm[b] * nzvalVm[c] * 
                                    (  nzvalYtfR[c] * cos(cuVa[b] - nzvalVa[c]) 
                                     + nzvalYtfI[c] * sin(cuVa[b] - nzvalVa[c])
                                    ) 
        end
      end
      return nothing
  end

  function cpu_term3(viewToI, cuVm, colptrVm, nzvalVm, colptrYftI, nzvalYftI,
                              cuVa, colptrVa, nzvalVa, colptrYftR, nzvalYftR,
                              sizeFromLines) 
      for b in 1:size(viewToI,1)
        for (i,c) in enumerate(colptrVm[b]:colptrVm[b+1]-1)
          viewToI[b] += cuVm[b] * nzvalVm[c] * 
                                  (  ( -nzvalYftI[c]) * cos(cuVa[b] - nzvalVa[c]) 
                                      + nzvalYftR[c] * sin(cuVa[b] - nzvalVa[c])
                                  ) 
        end
      end
      return nothing
  end

  function cpu_term4(viewFromI, cuVm, colptrVm, nzvalVm, colptrYtfI, nzvalYtfI,
                                  cuVa, colptrVa, nzvalVa, colptrYtfR, nzvalYtfR,
                                  sizeToLines) 
      for b in 1:size(viewFromI,1)
        for (i,c) in enumerate(colptrVm[b]:colptrVm[b+1]-1)
          viewFromI[b] += cuVm[b] * nzvalVm[c] * 
                                    (  ( -nzvalYtfI[c]) * cos(cuVa[b] - nzvalVa[c]) 
                                        + nzvalYtfR[c] * sin(cuVa[b] - nzvalVa[c])
                                    ) 
        end
      end
      return nothing
  end
  function cpu_sumPg(sumPg, colptr, nzval)
      for b in 1:size(sumPg,1)
        for c in colptr[b]:colptr[b+1]-1
            @inbounds sumPg[b] += nzval[c]
        end
      end
      return nothing
  end
  function cpu_sumQg(sumQg, colptr, nzval)
      for b in 1:size(sumQg,1)
        for c in colptr[b]:colptr[b+1]-1
            sumQg[b] += nzval[c]
        end
      end
      return nothing
  end
end