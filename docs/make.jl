using fermi_cg
using Documenter

makedocs(;
    modules=[fermi_cg],
    authors="Nick Mayhall <nmayhall@vt.edu> and contributors",
    repo="https://github.com/nmayhall/fermi_cg.jl/blob/{commit}{path}#L{line}",
    sitename="fermi_cg.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nmayhall.github.io/fermi_cg.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nmayhall/fermi_cg.jl",
)
