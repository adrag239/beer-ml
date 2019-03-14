﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace BeerML.Clustering
{
    public class ClusteringData
    {
        [LoadColumn(0)]
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
            var mlContext = new MLContext();


            // Load training data
            var trainingDataView = mlContext.Data.LoadFromTextFile<ClusteringData>(
              "4_Clustering/problem4.csv",
              hasHeader: true,
              separatorChar: ',');

            // Define features
            var dataProcessPipeline =
                    mlContext.Transforms.Text.FeaturizeText("FullNameFeaturized", "FullName")
                    .Append(mlContext.Transforms.Concatenate("Features", "FullNameFeaturized"));

            // Use KMeans clustering
            var trainer = mlContext.Clustering.Trainers.KMeans(new KMeansPlusPlusTrainer.Options
                {
                    ClustersCount = 2,
                    FeatureColumn = "Features"
                }
            );

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model based on training data
            var watch = Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();

            Console.WriteLine($"Trained the model in: {watch.ElapsedMilliseconds / 1000} seconds.");

            var predFunction = trainedModel.CreatePredictionEngine<ClusteringData, ClusteringPrediction>(mlContext);

            // Use model
            IEnumerable<ClusteringData> drinks = new[]
            {
                new ClusteringData { FullName = "Inchgower Maltbarn Bourbon Cask 24 Years" },
                new ClusteringData { FullName = "Caol Ila 9 Years"},
                new ClusteringData { FullName = "Ardmore 21 Years"},
                new ClusteringData { FullName = "Crémant d'Alsace Riesling Brut"},
                new ClusteringData { FullName = "Charles Ellner Réserve Brut"},
                new ClusteringData { FullName = "De Vergy Prestige Rosé"}
            };

            foreach (var drink in drinks)
            {
                var prediction = predFunction.Predict(drink);

                Console.WriteLine($"{drink.FullName} is {prediction.SelectedClusterId}");
            }

        }
    }
}
