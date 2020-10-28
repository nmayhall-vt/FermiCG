using FermiCG
using Documenter

makedocs(;
    modules=[FermiCG],
    authors="Nick Mayhall <nmayhall@vt.edu> and contributors",
    repo="https://github.com/nmayhall/FermiCG.jl/blob/{commit}{path}#L{line}",
    sitename="FermiCG.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nmayhall.github.io/FermiCG.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nmayhall/FermiCG.jl",
)
