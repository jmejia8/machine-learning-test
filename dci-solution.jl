include("tools.jl")

function main()
    nruns = 31

    X, y, X_test, y_test, X_validate = getData()

    accuracy_list = Real[]

    model = RandomForestClassifier(n_estimators=200, max_features=10, n_jobs=4, criterion="entropy")

    for i = 1:nruns

        # train the model
        fit!(model, X, y)

        # predict the classes for the testing set
        y_approx = predict(model, X_test)

        # prediction accuracy
        accuracy = 100sum(y_approx .== y_test) / length(y_test)
        
        push!(accuracy_list, accuracy)

        @printf("run: %02d \t accuracy: %.2f\n", i, accuracy )
    end

    println("\nResults of 31 independent runs.")
    
    @printf("Best accuracy:\t%.2f\n", maximum(accuracy_list))
    @printf("Accuracy mean:\t%.2f\n", mean(accuracy_list))
    @printf("Accuracy  std:\t%.2f\n", std(accuracy_list))
    @printf("Worst accuracy:\t%.2f\n", minimum(accuracy_list))

    # save the classes of validate set.
    fit!(model, X, y)
    y_approx = predict(model, X_validate)

    writedlm("retoFiltro.txt", floor.(Int, y_approx), '\n')

end

main()

# output
# run: 01      accuracy: 91.14
# run: 02      accuracy: 91.92
# run: 03      accuracy: 91.45
# run: 04      accuracy: 91.53
# run: 05      accuracy: 91.06
# run: 06      accuracy: 91.14
# run: 07      accuracy: 91.61
# run: 08      accuracy: 91.45
# run: 09      accuracy: 91.53
# run: 10      accuracy: 91.84
# run: 11      accuracy: 91.76
# run: 12      accuracy: 91.61
# run: 13      accuracy: 91.61
# run: 14      accuracy: 92.15
# run: 15      accuracy: 91.61
# run: 16      accuracy: 91.69
# run: 17      accuracy: 91.53
# run: 18      accuracy: 91.30
# run: 19      accuracy: 91.84
# run: 20      accuracy: 91.61
# run: 21      accuracy: 91.53
# run: 22      accuracy: 91.38
# run: 23      accuracy: 91.69
# run: 24      accuracy: 90.83
# run: 25      accuracy: 91.76
# run: 26      accuracy: 91.61
# run: 27      accuracy: 91.38
# run: 28      accuracy: 91.76
# run: 29      accuracy: 92.00
# run: 30      accuracy: 91.53
# run: 31      accuracy: 91.14

# Results of 31 independent runs.
# Best accuracy:  92.15
# Accuracy mean:  91.55
# Accuracy  std:  0.29
# Worst accuracy: 90.83
