
module ad
    using Calculus
    export MyNumber
    export differentiate
    
    macro ad(ex)
        @show a
    end

    struct MyNumber
        v::Float64
        d::Float64
    end

    # Unary
    
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

    
    # Binary
end

using .ad