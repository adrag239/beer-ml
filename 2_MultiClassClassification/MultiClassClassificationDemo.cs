using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BeerML.MultiClassClassification
{
    public class DrinkData
    {
        [LoadColumn(0)]
        public string FullName;
        [LoadColumn(1)]
        public string Type;
        [LoadColumn(2)]
        public string Country;
    }

    public class DrinkPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Type;

        [ColumnName("Score")]
        public float[] Scores;
    }

    public class MultiClassClassificationDemo
    {
        public static void Run()
        {
            // Define context
            var mlContext = new MLContext();

            // Load training data
            var trainingDataView = mlContext.Data.LoadFromTextFile<DrinkData>(
               "2_MultiClassClassification/problem2_train.csv",
               hasHeader: true,
               separatorChar: ',');

            // Define features
            var dataProcessPipeline = 
                mlContext.Transforms.Conversion.MapValueToKey("Label", "Type")
                    .Append(mlContext.Transforms.Text.FeaturizeText("FullNameFeaturized", "FullName"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("CountryEncoded", "Country"))
                    .Append(mlContext.Transforms.Concatenate("Features", "FullNameFeaturized", "CountryEncoded"));

            // Use Multiclass classification
            var trainer = mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features);

            var trainingPipeline = dataProcessPipeline
                .Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model based on training data
            var watch = Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();

            Console.WriteLine($"Trained the model in: {watch.ElapsedMilliseconds / 1000} seconds.");

            // Use model for predictions
            IEnumerable<DrinkData> drinks = new[]
            {
                new DrinkData { FullName = "Weird Stout" },
                new DrinkData { FullName = "Folkes Röda IPA"},
                new DrinkData { FullName = "Fryken Havre Ale"},
                new DrinkData { FullName = "Barolo Gramolere"},
                new DrinkData { FullName = "Château de Lavison"},
                new DrinkData { FullName = "Korlat Cabernet Sauvignon"},
                new DrinkData { FullName = "Glengoyne 25 Years"},
                new DrinkData { FullName = "Oremus Late Harvest Tokaji Cuvée"},
                new DrinkData { FullName = "Izadi Blanco"},
                new DrinkData { FullName = "Ca'Montini Prosecco Extra Dry"}
            };

            var predFunction = trainedModel.CreatePredictionEngine<DrinkData, DrinkPrediction>(mlContext);

            foreach (var drink in drinks)
            {
                var prediction = predFunction.Predict(drink);

                Console.WriteLine($"{drink.FullName} is {prediction.Type}");
            }

            // Evaluate the model
            var testDataView = mlContext.Data.LoadFromTextFile<DrinkData>(
               "2_MultiClassClassification/problem2_validate.csv",
               hasHeader: true,
               separatorChar: ',');
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy: {metrics.AccuracyMicro:P2}");

        }
    }
}
