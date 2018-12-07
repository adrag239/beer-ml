using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;

namespace BeerML.Clustering
{
    public class ClusteringData
    {
        [Column(ordinal: "0")]
        public string FullName;
    }
    public class ClusteringPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint SelectedClusterId;
        [ColumnName("Score")]
        public float[] Distance;
    }

    public class ClusteringDemo
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
                }
            });

            // Load training data
            var trainingDataView = textLoader.Read("4_Clustering/problem4.csv");

            // Define features
            var dataProcessPipeline =
                    mlContext.Transforms.Text.FeaturizeText("FullName", "FullNameFeaturized")
                    .Append(mlContext.Transforms.Concatenate("Features", "FullNameFeaturized"));

            // Use KMeans clustering
            var trainer = mlContext.Clustering.Trainers.KMeans(features: "Features", clustersCount: 2);

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model based on training data
            var watch = Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();

            Console.WriteLine($"Trained the model in: {watch.ElapsedMilliseconds / 1000} seconds.");

            var predFunction = trainedModel.MakePredictionFunction<ClusteringData, ClusteringPrediction>(mlContext);

            // Use model
            IEnumerable<ClusteringData> drinks = new[]
            {
                new ClusteringData { FullName = "Inchgower Maltbarn Bourbon Cask 24 Years" },
                new ClusteringData { FullName = "Caol Ila 9 Years"},
                new ClusteringData { FullName = "Ardmore 21 Years"},
                new ClusteringData { FullName = "Crémant d'Alsace Riesling Brut"},
                new ClusteringData { FullName = "Laurent-Perrier Millesime"},
                new ClusteringData { FullName = "Lellè Prosecco Spumante"}
            };

            foreach (var drink in drinks)
            {
                var prediction = predFunction.Predict(drink);

                Console.WriteLine($"{drink.FullName} is {prediction.SelectedClusterId}");
            }

        }
    }
}
