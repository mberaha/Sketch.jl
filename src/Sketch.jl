module Sketch

include("data.jl")
include("estimate_params.jl")
include("multiview.jl")
include("params.jl")
include("smoothed_estimators.jl")
include("utils.jl")

greet() = print("Hello World!")

end # module sketch
