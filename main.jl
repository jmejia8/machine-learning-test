import DelimitedFiles.readdlm
import Random.randperm
using PyPlot
using RDatasets: dataset
using ScikitLearn
import Statistics: mean, std
import Printf.@printf
import MultivariateStats


######################################################################
#
# Loading the algorithms
#
######################################################################
if !@isdefined(StandardScaler)
    @sk_import preprocessing: StandardScaler
    @sk_import datasets: (make_moons, make_circles, make_classification)
    @sk_import neighbors: KNeighborsClassifier
    @sk_import svm: SVC
    @sk_import tree: DecisionTreeClassifier
    @sk_import ensemble: (RandomForestClassifier, AdaBoostClassifier)
    @sk_import naive_bayes: GaussianNB
    @sk_import discriminant_analysis: (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
end


clf_names = ["Nearest Neighbors",
             "Linear SVM",
             "RBF SVM",
             "Decision Tree",
             "Random Forest",
             "AdaBoost",
             "Naive Bayes",
             "Linear Discriminant Analysis",
             "Quadratic Discriminant Analysis"]

classifiers = [
    KNeighborsClassifier(3; p=1, algorithm="auto", weights="distance"),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=200),
    RandomForestClassifier(n_estimators=200, max_features=10, n_jobs=4, criterion="entropy"),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]

######################################################################


function getPCA(X, maxdim = 10)
    # train a PCA model
    model = MultivariateStats.fit(MultivariateStats.PCA, X'; maxoutdim=maxdim)

    # apply PCA model to testing set
    A = MultivariateStats.transform(model, X')

    return convert(Array, A')

end


function getData(usePCA = false)
    trainSet = readdlm("data/trainSet.csv", ',', Float64, '\n')
    testSet  = readdlm("data/testSet.csv", ',', Float64, '\n')
    X_validate  = readdlm("data/validateSet.csv", ',', Float64, '\n')


    X = trainSet[:,1:end-1]
    y = trainSet[:,end]


    X_test = testSet[:,1:end-1]
    y_test = testSet[:,end]

    if usePCA
        X = getPCA(X)
        X_test = getPCA(X_test)
        X_validate = getPCA(X_validate)
    end

    return X, y, X_test, y_test, X_validate
end

function plotScatter(model_k)

    println(model_k)
    model = classifiers[model_k]

    X_train, y_train, X, y, X_validate = getData()

    fit!(model, X_train, y_train)

    # find the classes for the validate set.
    y_validate = predict(model, X_validate)
    y_approx = predict(model, X)

    if size(X_train, 2) > 2 
        X_train = getPCA(X_train, 2)
    end

    if size(X, 2) > 2 
        X = getPCA(X, 2)
    end

    if size(X_validate, 2) > 2 
        X_validate = getPCA(X_validate, 2)
    end

    i = y .== y_approx

    labels = Set(y)

    subplot(2, 2, 1)
    title("Train Set")
    for label = labels
        plot(X_train[y_train .== label,1], X_train[y_train .== label,2], marker=:o, lw=0)
    end
    xlim([-200, 200]); ylim([-300, 150])


    subplot(2, 2, 2)
    title("Test Set: Exact classes")
    for label = labels
        plot(X[y .== label,1], X[y .== label,2], marker=:o, lw=0)
    end
    xlim([-200, 200]); ylim([-300, 150])

    subplot(2, 2, 3)
    title("Test set: Approximate classes")
    for label = labels
        plot(X[y_approx .== label,1], X[y_approx .== label,2], marker=:o, lw=0)
    end
    xlim([-200, 200]); ylim([-300, 150])

    subplot(2, 2, 4)
    title("Validate set: Approximate classes")
    for label = labels
        plot(X_validate[y_validate .== label,1], X_validate[y_validate .== label,2], marker=:o, lw=0)
    end
    xlim([-200, 200]); ylim([-300, 150])

end

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
        @printf("%s: \t %.2f \t std = %f\n", name, 100*mean(accuracy_list), std(accuracy_list))

        k += 1
        # break
    end

    # plot the results
    plotScatter(best_model)


end

test()