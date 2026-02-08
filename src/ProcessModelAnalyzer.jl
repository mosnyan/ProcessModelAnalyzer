module ProcessModelAnalyzer
    using DifferentialEquations
    using CSV
    using DataFrames
    using Interpolations
    using Optimization
    using OptimizationOptimJL
    using OptimizationBBO

    export optimize_model

    @enum RecommendedStrategy begin
        Advanced
        PID
        PI
        P
        BangBang
    end

    """
        optimize_model(file_path)

    Runs an optimizer on .csv process data. The column format must be : 1. time, 2. CV, 3. PV
    """
    function optimize_model(file_path)
        data = set_up_data(file_path)

        initial_vector = [1.0, 1.0, 1.0]
        lower_bounds = [-20.0, 0.0, 0.0]
        upper_bounds = [20.0, 20.0, 20.0]

        black_box_function = OptimizationFunction(loss_function)
        black_box_problem = OptimizationProblem(black_box_function, initial_vector, data; lb=lower_bounds, ub=upper_bounds)
        black_box_solution = solve(black_box_problem, BBO_adaptive_de_rand_1_bin_radiuslimited())

        refined_function = OptimizationFunction(loss_function, ADTypes.AutoForwardDiff())
        refined_problem = OptimizationProblem(refined_function, black_box_solution.minimizer, data; lb=lower_bounds, ub=upper_bounds)
        results = solve(refined_problem, BFGS())

        strategy = recommend_strategy(results.u[2], results.u[3])

        return (
            residual_error = results.objective,
            kp = results.u[1],
            τ = results.u[2],
            θ = results.u[3],
            recommended_strategy = strategy
        )
    end

    """
        first_order!(dy, y, p, t)

    In-place version of the first-order plus dead time model differential equation.
    """
    function first_order!(dy, y, p, t)
        u_delayed = t < p.θ ? p.bias_u : p.interpolated_u(t - p.θ)
        dy[1] = (p.kp * (u_delayed - p.bias_u) - (y[1] - p.bias_y)) / p.τ
    end

    """
        generate_model(diff_function!, p)

    Calculates the model differential equation over the data time span.
    """
    function generate_model(diff_function!, p)
        ode_problem = ODEProblem(diff_function!, [p.bias_y], p.t_span, p)
        return solve(ode_problem; abstol=10e-6, reltol=10e-6)
    end

    """
        calculate_model_sse(model, data)

    Calculates the sum of squared errors of the model versus the data.
    """
    function calculate_model_sse(model, data)
        if SciMLBase.successful_retcode(model)
            model_y = model.(data.sampled_t; idxs=1)
            return sum((model_y .- data.sampled_y) .^ 2)
        end

        error("DiffEq solver has failed with code " * model.retcode)
    end

    """
        loss_function(parameter_vector, data)

    Sets up the optimizer with the loss function.
    """
    function loss_function(parameter_vector, data)
        model_parameters = (
            kp = parameter_vector[1],
            τ = parameter_vector[2],
            θ = parameter_vector[3]
        )

        model = generate_model(first_order!, merge(model_parameters, data))
        error = calculate_model_sse(model, data)
        return error
    end

    """
        set_up_data(file_path)

    Reads the CSV, and sets up the data needed for the fit.
    """
    function set_up_data(file_path)
        file = CSV.read(open(file_path), DataFrame; header=false)
        return (
            sampled_t = file.Column1 .- file.Column1[1],
            sampled_u = file.Column2,
            interpolated_u = linear_interpolation(file.Column1, file.Column2, extrapolation_bc=Line()),
            sampled_y = file.Column3,
            bias_y = file.Column3[1],
            bias_u = file.Column2[1],
            t_span = (0.0, file.Column1[end])
        )
    end

    """
        recommend_strategy(τ, θ)

    Calculates the τ/θ controllability ratio and suggests a strategy.
    """
    function recommend_strategy(τ, θ)
        if θ <= 0
            error("θ cannot be 0.")
        end

        ratio = τ / θ

        if 1 >= ratio
            return Advanced
        elseif 1 < ratio
            return PID
        elseif 2 < ratio
            return PI
        else
            return P
        end
    end

end # module ProcessModelAnalyzer