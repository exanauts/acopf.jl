using Revise

using Ipopt

using JuMP

using Test

include("powerad.jl")

model = Model(with_optimizer(Ipopt.Optimizer))
@variable(model, x[1:4])
@NLconstraint(model, sum(x[i]^2 for i in 1:4) == 40.0)
@NLobjective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
optimize!(model)
@show sol = value.(x)



x = [1.0, 5.0, 5.0, 1.0]
x1 = 1.0
x2 = 5.0
x3 = 5.0
x4 = 1.0
function objective(x)
    return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
end

function constraint(x)
    return sum(x[i]^2 for i in 1:4) - 40
end

# ad.@objective("x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]", [:x])
# ad.@objective("x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]", [:x[1], :x[2], :x[3], x[4]])
obj = ad.@objective("x1 * x4 * (x1 + x2 + x3) + x3", [:x1, :x2, :x3, :x4])
con = ad.@objective("x1^2 + x2^2 + x3^2 + x4^2 - 40", [:x1, :x2, :x3, :x4])

hessian = Array{Int64,2}(undef, 4, 4)

# hessian .= getindex(hessian,1)
# @show eval_hess =  eval.(obj.hessian)

for el in obj.hessian
    @show eval.(el)
end

(x -> eval.(x)).(obj.hessian)



# @show hessian





