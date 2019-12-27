
module ad
    using Calculus
    export differentiate
    export @objective, @constraint 
    
    struct objexpr
        func
        jacobian
        hessian
        vars
    end

    struct conexpr
        func
        jacobian
        hessian
        symb
        rhs
        vars
    end

    function differentiate(ex::String, vars::Array{Symbol,1})
        Calculus.differentiate(ex, vars)
    end

    function differentiate(ex::String, vars::Symbol)
        Calculus.differentiate(ex, vars)
    end

    function differentiate(ex::Array{Expr,1}, vars::Array{Symbol,1})
        ret = []
        for el in ex
            push!(ret, Calculus.differentiate(el, vars))
        end
        ret
    end

    function differentiate(ex::Array{Any,1}, vars::Array{Symbol,1})
        ret = []
        for el in ex
            push!(ret, Calculus.differentiate(el, vars))
        end
        ret
    end

    macro objective(func, vars)
        jac = ad.differentiate(func, eval(vars))
        hes = ad.differentiate(jac, eval(vars))
        ret = objexpr(func, jac, hes, vars)
    end

    macro constraint(func, symb, rhs, vars)
        jac = ad.differentiate(func, eval(vars))
        hes = ad.differentiate(jac, eval(vars))
        ret = conexpr(func, jac, hes, symb, rhs, vars)
    end
end

using .ad

obj = @objective("cos(x)*sin(y)", [:x, :y])
con = @constraint("cos(x)*sin(y)", "==", "0", [:x, :y])

x = 1.0
y = 3.14

eval(obj.hessian[1][1])
