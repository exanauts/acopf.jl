include("powerad.jl")
using .ad

a = :(x1*x2)
b = differentiate(a)
@show typeof(b)
printast(a)
printast(b)

using CuArrays
n = Int(1e1)
array = Array{Float64,1}(undef, n) # 7 GB
cuarray = CuArray{Float64,1,Nothing}(undef, n)
# Classic loop, not parallel 
for el in array
  el^2
end
# broadcast notation using CPU threads
array.^2
# Fast broadcast on GPU
cuarray.^2

using TimerOutputs
timeroutput = TimerOutput()

using CuArrays
array = Array{Float64,1}(undef, n)
cuarray = CuArray{Float64,1,Nothing}(undef, n)
@timeit timeroutput "loop" begin
for el in array
  el^2
end
end
# broadcast notation
@timeit timeroutput "CPU" begin
array.^2
end
# Fast broadcast on GPU
@timeit timeroutput "GPU" begin
cuarray.^2
end
show(timeroutput)

x = 1
y = ones(n) # 7 GB
@show y

z = x .- y

@show z
