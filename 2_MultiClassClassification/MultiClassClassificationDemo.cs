using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace BeerML.MultiClassClassification
{
    public class DrinkData
    {
        [Column(ordinal: "0")]
        public string FullName;
        [Column(ordinal: "1")]
        public string Type;
        [Column(ordinal: "2")]
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
            // Define pipeline
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader("2_MultiClassClassification/problem2_train.csv").CreateFrom<DrinkData>(useHeader: true, separator: ','));

            pipeline.Add(new TextFeaturizer("FullName", "FullName"));
            pipeline.Add(new TextFeaturizer("Country", "Country"));
            pipeline.Add(new ColumnConcatenator("Features", "FullName", "Country"));

            pipeline.Add(new Dictionarizer(("Type", "Label")));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier { });

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // Train model
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var model = pipeline.Train<DrinkData, DrinkPrediction>();
            stopWatch.Stop();
            Console.WriteLine($"Trained the model in: {stopWatch.ElapsedMilliseconds / 1000} seconds.");

            // Evaluate model
            var testData = new TextLoader("2_MultiClassClassification/problem2_validate.csv").CreateFrom<DrinkData>(useHeader: true, separator: ',');

            var evaluator = new ClassificationEvaluator { OutputTopKAcc = 1 };
            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine(metrics.TopKAccuracy.ToString("P"));

            // Use model
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

            string[] names;
            model.TryGetScoreLabelNames(out names);

            var predictions = model.Predict(drinks).ToList();

        }
    }
}
