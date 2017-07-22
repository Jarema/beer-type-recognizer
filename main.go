package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	rawData, err := base.ParseCSVToInstances("beers.csv", true)
	if err != nil {
		panic(err)
	}
	fmt.Println(rawData)

	cls := knn.NewKnnClassifier("euclidean", "linear", 1)

	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.80)

	fmt.Println(trainData, testData)
	cls.Fit(trainData)

	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}

	fmt.Println("predicitons:\n", predictions)

	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))

	exampleData, err := base.ParseCSVToTemplatedInstances("example.csv", true, rawData)
	if err != nil {
		panic(err)
	}
	check, err := cls.Predict(exampleData)
	if err != nil {
		panic(err)
	}

	fmt.Println("check:", check)

}
