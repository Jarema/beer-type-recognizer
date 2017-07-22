package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
	"io/ioutil"
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

	exampleFile, err := ioutil.ReadFile("example.csv")
	if err != nil {
		panic(err)
	}
	r := csv.NewReader(bytes.NewReader(exampleFile))
	records, err := r.ReadAll()
	if err != nil {
		panic(err)
	}

	_, size := check.Size()
	for i := 0; i < size; i++ {
		headers := records[0]
		fmt.Printf("%v:%v, %v:%v, type: %v\n", headers[0], records[i+1][0], headers[1], records[i+1][1], check.RowString(i))
	}

}
