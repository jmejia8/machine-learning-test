include("tools.jl")

function test(plotResult = false)
    # loading the data
    X, y, X_test, y_test, X_validate = getData()


    best_accu = -Inf
    best_model = -1

    k = 1
    for (name, clf) in zip(clf_names, classifiers)
        model = clf

        accuracy_list = Real[]

        y_approx = nothing
        for i = 1:5
            fit!(model, X, y)
            y_approx = predict(model, X_test)
            accuracy = sum(y_approx .== y_test) / length(y_test)
            push!(accuracy_list, accuracy)
        end

        if best_accu < mean(accuracy_list)
            best_model = k
            best_accu = mean(accuracy_list)
        end

        # Print stats
        @printf("%s: \t accu. = %.2f \t std = %f\n", name, 100*mean(accuracy_list), std(accuracy_list))

        k += 1
        # break
    end


    if plotResult
        # plot the results
        plotScatter(best_model)
    end

    println("\n\nThe best classifier is: ", clf_names[best_model], " with accuracy ", 100*best_accu)

end

test(true)