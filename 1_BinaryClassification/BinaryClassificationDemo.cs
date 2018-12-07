using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;

namespace BeerML.BinaryClassification
{
    public class BeerOrWineData
    {
        [Column(ordinal: "0")]
        public string FullName;
        [Column(ordinal: "1")]
        public bool Beer;
    }

    public class BeerOrWinePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Beer;

        public float Probability { get; set; }

        public float Score { get; set; }
    }

    public class BinaryClassificationDemo
    {
        public static void Run()
        {
            // Define context
            var mlContext = new MLContext(seed: 0);

            // Define data file format
            TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("FullName", DataKind.Text, 0),
                    new TextLoader.Column("Beer", DataKind.Bool, 1)
                }
            });

            // Load training data
            var trainingDataView = textLoader.Read("1_BinaryClassification/problem1_train.csv");

            // Define features
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("FullName", "Features");

            // Use Binary classification
            var trainer = mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Beer", featureColumn: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model based on training data
            var watch = Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();

            Console.WriteLine($"Trained the model in: {watch.ElapsedMilliseconds / 1000} seconds.");

            // Use model for predictions
            List<BeerOrWineData> drinks = new List<BeerOrWineData>
            {
                new BeerOrWineData { FullName = "Castle Stout" },
                new BeerOrWineData { FullName = "Folkes Röda IPA"},
                new BeerOrWineData { FullName = "Fryken Havre Ale"},
                new BeerOrWineData { FullName = "Barolo Gramolere"},
                new BeerOrWineData { FullName = "Château de Lavison"},
                new BeerOrWineData { FullName = "Korlat Cabernet Sauvignon"}
            };

            var predFunction = trainedModel.MakePredictionFunction<BeerOrWineData, BeerOrWinePrediction>(mlContext);

            foreach (var drink in drinks)
            {
                var prediction = predFunction.Predict(drink);

                Console.WriteLine($"{drink.FullName} is {prediction.Beer}");
            }

            // Evaluate the model
            var testDataView = textLoader.Read("1_BinaryClassification/problem1_validate.csv");
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Beer", "Score");

            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");

            // Cross validation
            var fullDataView = textLoader.Read("1_BinaryClassification/problem1.csv");
            var cvResults = mlContext.BinaryClassification.CrossValidate(fullDataView, trainingPipeline, numFolds: 5, labelColumn: "Beer");
            Console.WriteLine($"Avg Accuracy is: {cvResults.Select(r => r.metrics.Accuracy).Average():P2}");
        }
    }
}
