
function test1(v)
    resize!(v, length(v)+2)
    v = FermiCG.reshape2(v, (length(v)÷2, 2))
end

function test2(v)
    test1(v)
end

function test3(v)
    test2(v)
end

A = rand(10000)
println(typeof(A), pointer(A))
@time test1(A)
println(typeof(A), pointer(A))
@time test2(A)
println(typeof(A), pointer(A))
@time test3(A)
println(typeof(A), pointer(A))