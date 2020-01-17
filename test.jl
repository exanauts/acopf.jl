using Revise

using Ipopt

using JuMP

using Test
using Printf
using SymEngine

include("powerad.jl")

# model = Model(with_optimizer(Ipopt.Optimizer))
# @variable(model, x[1:4])
# @NLconstraint(model, sum(x[i]^2 for i in 1:4) == 40.0)
# @NLobjective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
# set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
# optimize!(model)
# @show sol = value.(x)


function printast(expr,n)
    for i in 1:n @printf("\t") end
    if typeof(expr) == Symbol
        @printf("Symbol: %s\n", expr)
        return
    end
    if typeof(expr) == Int64
        @printf("Int64: %s\n", expr)
        return
    end
    if typeof(expr) == Float64
        @printf("Float64: %s\n", expr)
        return
    end
    @printf("%s\n", expr.head)
    for i in expr.args
        printast(i, n+1)
    end
end

function printast(expr)
    @show "Showing expression", expr
    printast(expr,0)
end

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

# ad.@objective("x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]", [:x[1], :x[2], :x[3], x[4]])
obj = ad.@objective("x1 * x4 * (x1 + x2 + x3) + x3", [:x1, :x2, :x3, :x4])
con = ad.@objective("x1^2 + x2^2 + x3^2 + x4^2 - 40", [:x1, :x2, :x3, :x4])

hessian = Array{Expr,2}(undef, 4, 4)

# hessian .= getindex(hessian,1)
# @show eval_hess =  eval.(obj.hessian)

for el in obj.hessian
    @show eval.(el)
end

(x -> eval.(x)).(obj.hessian)
a,b = symbols("a b")
t = 5.0

ex1 = t * a * b

diff(ex1,a)
x = [symbols("x[$i]") for i in 1:4]

for (i,el1) in enumerate(x)
    for (j,el2) in enumerate(x)
        hessian[i,j] = diff(diff(x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3], el1), el2)
    end
end
# x = [1.0, 5.0, 5.0, 1.0]
@show hessian[1,1]
printast(:((x1 * x2) + (x3 * x4) ))




# @show hessian





