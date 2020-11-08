using FermiCG
using Documenter

pages = [
    "Home" => "index.md",
    "Installation Instructions" => "installation_instructions.md",
    # "Code Basics" => "basics.md",
    # "Grids" => "grids.md",
    # "Problem" => "problem.md",
    # "GPU" => "gpu.md",
    "Examples" => ["cmf.md"],
    "Library" => [
        "Contents" => "library/outline.md",
        "Public" => "library/public.md",
        "Private" => "library/internals.md",
        "Function index" => "library/function_index.md",
        ],
]

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "test_cmf.jl",
]

# for example in examples
#   example_filepath = joinpath(EXAMPLES_DIR, example)
#   withenv("GITHUB_REPOSITORY" => "FourierFlows/FourierFlowsDocumentation") do
#     example_filepath = joinpath(EXAMPLES_DIR, example)
#     Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
#     Literate.notebook(example_filepath, OUTPUT_DIR, documenter=true)
#     Literate.script(example_filepath, OUTPUT_DIR, documenter=true)
#   end
# end

makedocs(
    modules=[FermiCG],
    authors="Nick Mayhall <nmayhall@vt.edu> and contributors",
    repo="https://github.com/nmayhall-vt/FermiCG/blob/{commit}{path}#L{line}",
    sitename="FermiCG.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nmayhall-vt.github.io/FermiCG",
        assets=String[],
    ),
    pages=pages,
)

deploydocs(
    repo="github.com/nmayhall-vt/FermiCG",
    push_preview = true,
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math")
)
