using FermiCG
using Test

@testset "strings" begin
    display(StringCI.string_to_index("110010"))
    @test StringCI.string_to_index("110010") == 19
end
