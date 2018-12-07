using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace BeerML.Regression
{
    public class PriceData
    {
        [Column(ordinal: "0")]
        public string FullName;
        [Column(ordinal: "1")]
        public float Price;
        [Column(ordinal: "2")]
        public float Volume;
        [Column(ordinal: "3")]
        public string Type;
        [Column(ordinal: "4")]
        public string Country;
    }

    public class PricePrediction
    {
        [ColumnName("Score")]
        public float Price;
    }

    public class RegressionDemo
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
                    new TextLoader.Column("Price", DataKind.R4, 1),
                    new TextLoader.Column("Volume", DataKind.R4, 2),
                    new TextLoader.Column("Type", DataKind.Text, 3),
                    new TextLoader.Column("Country", DataKind.Text, 4)
                }
            });

            // Load training data
            var trainingDataView = textLoader.Read("3_Regression/problem3_train.csv");

            // Define features
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Price", "Label")
                            .Append(mlContext.Transforms.Text.FeaturizeText("FullName", "FullNameFeaturized"))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Type", "TypeEncoded"))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Country", "CountryEncoded"))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Volume", "VolumeEncoded"))
                            .Append(mlContext.Transforms.Concatenate("Features", "FullNameFeaturized", "TypeEncoded", "CountryEncoded", "VolumeEncoded"));


            // Use Poisson Regressionn
            var trainer = mlContext.Regression.Trainers.PoissonRegression(labelColumn: "Label", featureColumn: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);


            // Train the model based on training data
            var watch = Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();

            Console.WriteLine($"Trained the model in: {watch.ElapsedMilliseconds / 1000} seconds.");

            // Use model for predictions
            IEnumerable<PriceData> drinks = new[]
            {
                new PriceData { FullName="Miami Vice Pale Ale", Type="Öl", Volume=330, Country="Danmark" },
                new PriceData { FullName="Hofbräu München Weisse", Type="Öl", Volume=500, Country="Tyskland" },
                new PriceData { FullName="Stefanus Blonde Ale", Type="Öl", Volume=330, Country="Belgien" },
                new PriceData { FullName="Mortgage 10 years", Type="Whisky", Volume=700, Country="Storbritannien" },
                new PriceData { FullName="Mortgage 21 years", Type="Whisky", Volume=700, Country="Storbritannien" },
                new PriceData { FullName="Merlot Classic", Type="Rött vin", Volume=750, Country="Frankrike" },
                new PriceData { FullName="Merlot Grand Cru", Type="Rött vin", Volume=750, Country="Frankrike" },
                new PriceData { FullName="Château de la Berdié Grand Cru", Type="Rött vin", Volume=750, Country="Frankrike" }
            };

            var predFunction = trainedModel.MakePredictionFunction<PriceData, PricePrediction>(mlContext);

            foreach (var drink in drinks)
            {
                var prediction = predFunction.Predict(drink);

                Console.WriteLine($"{drink.FullName} is {prediction.Price}");
            }

            // Evaluate the model
            // var testDataView = textLoader.Read("3_Regression/problem3_validate.csv");
            // var predictions = trainedModel.Transform(testDataView);
            // var metrics = mlContext.Regression.Evaluate(predictions, label: "Label", score: "Score");

        }

    }
}
