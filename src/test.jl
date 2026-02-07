using CSV
using DataFrames
using Interpolations
using DifferentialEquations

function get_data(file_path)
    file = CSV.read(open(file_path), DataFrame; header=false)
    return (
        sampled_t = file.Column1,
        sampled_u = file.Column2,
        interpolated_u = linear_interpolation(file.Column1, file.Column2, extrapolation_bc=Line()),
        sampled_y = file.Column3,
        bias_y = file.Column3[1],
        bias_u = file.Column2[1],
        t_span = (0.0, file.Column1[end])
    )
end

function fopdt(y, p, t)
    u_delayed = delay_input(p.data.interpolated_u, t, p.model_parameters.θ, p.data.bias_u)
    return (p.model_parameters.kp * (u_delayed - p.data.bias_u) - (y - p.data.bias_y)) / p.model_parameters.τ
end

function delay_input(u, t, θ, bias_u)
    return t < θ ? bias_u : u(t - θ)
end

function calculate_model(diff_eq_function, model_parameters, data)
    model_problem = ODEProblem(diff_eq_function, data.bias_y, data.t_span, (model_parameters = model_parameters, data = data))
    return solve(model_problem; abstol = 10e-9, reltol = 10e-9)
end

function calculate_model_sse(solution, data)
    model_y = solution.(data.sampled_t)
    error = sum((model_y .- data.sampled_y).^2)
    return error
end


function fopdt2(y, p, t)
    u_delayed = t < p.model_parameters.θ ? p.data.bias_u : p.data.interpolated_u(t - p.model_parameters.θ)
    dydt = (-(y - p.data.bias_y) + p.model_parameters.kp * (u_delayed - p.data.bias_u)) / p.model_parameters.τ
    return dydt
end

d = get_data("/data/prog/julia/ProcessModelAnalyzer/data/test_data.csv")
s = calculate_model(fopdt2, (kp = 1, τ = 1, θ = 1), d)
e = calculate_model_sse(s, d)