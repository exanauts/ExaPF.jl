
"""
    AbstractNLPEvaluator

AbstractNLPEvaluator implements the bridge between the
problem formulation (see `AbstractFormulation`) and the optimization
solver. Once the problem formulation bridged, the evaluator allows
to evaluate in a straightfoward fashion the objective and the different
constraints, but also the corresponding gradient and Jacobian objects.

"""
abstract type AbstractNLPEvaluator end

abstract type AbstractNLPAttribute end

"""
    Variables <: AbstractNLPAttribute end

Attribute corresponding to the optimization variables attached
to a given `AbstractNLPEvaluator`.
"""
struct Variables <: AbstractNLPAttribute end

"""
    Constraints <: AbstractNLPAttribute end

Attribute corresponding to the constraints  attached
to a given `AbstractNLPEvaluator`.
"""
struct Constraints <: AbstractNLPAttribute end

"""
    AutoDiffBackend <: AbstractNLPAttribute end

Attribute corresponding to the autodiff backend used
inside the `AbstractNLPEvaluator`.
"""
struct AutoDiffBackend <: AbstractNLPAttribute end

"""
    n_variables(nlp::AbstractNLPEvaluator)
Get the number of variables in the problem.
"""
function n_variables end

"""
    n_constraints(nlp::AbstractNLPEvaluator)
Get the number of constraints in the problem.
"""
function n_constraints end

# Callbacks
"""
    objective(nlp::AbstractNLPEvaluator, u)::Float64

Evaluate the objective at point `u`.
"""
function objective end

"""
    gradient!(nlp::AbstractNLPEvaluator, g, u)

Evaluate the gradient of the objective at point `u`. Store
the result inplace in the vector `g`, which should have the same
dimension as `u`.

"""
function gradient! end

"""
    constraint!(nlp::AbstractNLPEvaluator, cons, u)

Evaluate the constraints of the problem at point `u`. Store
the result inplace, in the vector `cons`.

## Note
The vector `cons` should have the same dimension as the result
returned by `n_constraints(nlp)`.

"""
function constraint! end

"""
    jacobian_structure!(nlp::AbstractNLPEvaluator, rows, cols)

Return the sparsity pattern of the Jacobian matrix.
"""
function jacobian_structure! end

"""
    jacobian!(nlp::ReducedSpaceEvaluator, jac, u)

Evaluate the Jacobian of the constraints ``J`` at position `u`. Store
the result inplace, in the `m x n` matrix `jac`.
"""
function jacobian! end

"""
    jtprod!(nlp::ReducedSpaceEvaluator, jv, u, v)

Evaluate the transpose Jacobian-vector product ``J^{T} v`` of the constraints.
The vector `jv` is modified inplace.

Let `(n, m) = n_variables(nlp), n_constraints(nlp)`.

* `u` is a vector with dimension `n`
* `v` is a vector with dimension `m`
* `jv` is a vector with dimension `n`

"""
function jtprod! end

"""
    ojtprod!(nlp::ReducedSpaceEvaluator, jv, u, σ, v)

Evaluate the transpose Jacobian-vector product `J' * [σ ; v]`,
with `J` the Jacobian of the vector `[f(x); h(x)]`.
`f(x)` is the current objective and `h(x)` constraints.
The vector `jv` is modified inplace.

Let `(n, m) = n_variables(nlp), n_constraints(nlp)`.

* `jv` is a vector with dimension `n`
* `u` is a vector with dimension `n`
* `σ` is a scalar
* `v` is a vector with dimension `m`

"""
function ojtprod! end

"""
    hessian!(nlp::AbstractNLPEvaluator, hess, u)

Evaluate the Hessian of the problem at point `u`. Store
the result inplace, in the vector `hess`.

"""
function hessian! end

# Utilities
"""
    primal_infeasibility(nlp::AbstractNLPEvaluator, u)

Return primal infeasibility associated to current model `nlp` evaluated
at control `u`.

"""
function primal_infeasibility end

"""
    primal_infeasibility!(nlp::AbstractNLPEvaluator, cons, u)

Return primal infeasibility associated to current model `nlp` evaluated
at control `u`. Modify vector `cons` inplace.

"""
function primal_infeasibility! end

"Return `true` if the problem is constrained, `false` otherwise."
is_constrained(nlp::AbstractNLPEvaluator) = n_constraints(nlp) > 0

function constraints_type end

"""
    reset!(nlp::AbstractNLPEvaluator)

Reset evaluator `nlp` to default configuration.

"""
function reset! end

include("common.jl")

# Based evaluators
include("reduced_evaluator.jl")
include("slack_evaluator.jl")
include("proxal_evaluators.jl")

# Penalty evaluators
include("penalty.jl")
include("auglag.jl")

# Bridge with MOI
include("MOI_wrapper.jl")
include("optimizers.jl")

