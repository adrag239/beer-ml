using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BeerML.Regression
{
    public class PriceData
    {
        [LoadColumn(0)]
        public string FullName;
        [LoadColumn(1)]
        public float Price;
        [LoadColumn(2)]
        public float Volume;
        [LoadColumn(3)]
        public string Type;
        [LoadColumn(4)]
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
            Console.WriteLine("Regression");

            // Define context
            var mlContext = new MLContext();

            // Load training data
            var trainingDataView = mlContext.Data.LoadFromTextFile<PriceData>(
              "3_Regression/problem3_train.csv",
              hasHeader: true,
              separatorChar: ',');

            // Define features
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", "Price")
                            .Append(mlContext.Transforms.Text.FeaturizeText("FullNameFeaturized", "FullName"))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding("TypeEncoded", "Type"))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding("CountryEncoded", "Country"))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding("VolumeEncoded", "Volume"))
                            .Append(mlContext.Transforms.Concatenate("Features", "FullNameFeaturized", "TypeEncoded", "CountryEncoded", "VolumeEncoded"));


            // Use Poisson Regressionn
            var trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Label", featureColumnName: "Features");

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

            var predFunction = mlContext.Model.CreatePredictionEngine<PriceData, PricePrediction>(trainedModel);

            foreach (var drink in drinks)
            {
                var prediction = predFunction.Predict(drink);

                Console.WriteLine($"{drink.FullName} is {prediction.Price} SEK");
            }

            Console.WriteLine();
        }

    }
}
