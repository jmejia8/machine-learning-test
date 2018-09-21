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

@sk_import preprocessing: StandardScaler
@sk_import datasets: (make_moons, make_circles, make_classification)
@sk_import neighbors: KNeighborsClassifier
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import ensemble: (RandomForestClassifier, AdaBoostClassifier)
@sk_import naive_bayes: GaussianNB
@sk_import discriminant_analysis: (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)


clf_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
classifiers = [
    KNeighborsClassifier(3;p=1,algorithm="auto", weights="distance"),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]

######################################################################

# loading the data
train = readdlm("trainSet.csv", ',', Float64, '\n')
test = readdlm("testSet.csv", ',', Float64, '\n')


X = train[:,1:end-1]
y = train[:,end]


X_test = test[:,1:end-1]
y_test = test[:,end]


for (name, clf) in zip(clf_names, classifiers)
	model = clf

	accuracy_list = Real[]

	for i = 1:5
		fit!(model, X, y)
		accuracy = sum(predict(model, X_test) .== y_test) / length(y_test)
		push!(accuracy_list, accuracy)
	end
	@printf("%s: \t %.2f \t std = %f\n", name, 100*mean(accuracy_list), std(accuracy_list))
	break
end