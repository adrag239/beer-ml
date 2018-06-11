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

namespace BeerML.BinaryClassification
{
    public class BeerOrWineData
    {
        [Column(ordinal: "0")]
        public string FullName;
        [Column(ordinal: "1")]
        public string Type;
    }

    public class BeerOrWinePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Type;
    }
    
    public class BinaryClassificationDemo
    {
        public static void Run()
        {
            // Define pipeline
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader("1_BinaryClassification/problem1_train.csv").CreateFrom<BeerOrWineData>(useHeader: true, separator: ','));

            pipeline.Add(new TextFeaturizer("Features", "FullName"));

            pipeline.Add(new Dictionarizer(("Type", "Label")));

            pipeline.Add(new StochasticDualCoordinateAscentBinaryClassifier() { });
            //pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 25, NumTrees = 25, MinDocumentsInLeafs = 2 });   // up to 91% 
            //pipeline.Add(new FastForestBinaryClassifier() { NumLeaves = 25, NumTrees = 25, MinDocumentsInLeafs = 2 });  // 86%
            //pipeline.Add(new StochasticDualCoordinateAscentBinaryClassifier() { }); // 95%
            //pipeline.Add(new StochasticGradientDescentBinaryClassifier { }); // 92%

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // Train model
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var model = pipeline.Train<BeerOrWineData, BeerOrWinePrediction>();
            stopWatch.Stop();
            Console.WriteLine($"Trained the model in: {stopWatch.ElapsedMilliseconds / 1000} seconds.");

            // Evaluate model
            var testData = new TextLoader("1_BinaryClassification/problem1_validate.csv").CreateFrom<BeerOrWineData>(useHeader: true, separator: ',');

            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine(metrics.Accuracy.ToString("P"));
            // show matrix

            // Use model
            IEnumerable<BeerOrWineData> drinks = new[]
            {
                new BeerOrWineData { FullName = "Castle Stout" },
                new BeerOrWineData { FullName = "Folkes Röda IPA"},
                new BeerOrWineData { FullName = "Fryken Havre Ale"},
                new BeerOrWineData { FullName = "Barolo Gramolere"},
                new BeerOrWineData { FullName = "Château de Lavison"},
                new BeerOrWineData { FullName = "Korlat Cabernet Sauvignon"}
            };

            var predictions = model.Predict(drinks).ToList();

        }

        public static void CrossValidate()
        {
            // Define pipeline
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader("1_BinaryClassification/problem1.csv").CreateFrom<BeerOrWineData>(useHeader: true, separator: ','));

            pipeline.Add(new TextFeaturizer("Features", "FullName"));

            pipeline.Add(new Dictionarizer(("Type", "Label")));

            pipeline.Add(new StochasticDualCoordinateAscentBinaryClassifier() { });

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // Cross validation
            var cv = new CrossValidator().CrossValidate<BeerOrWineData, BeerOrWinePrediction>(pipeline);

            // show matrix
        }
    }
}

//{
//    KeepDiacritics = false,
//    KeepPunctuations = false,
//    TextCase = TextNormalizerTransformCaseNormalizationMode.Lower,
//    OutputTokens = true,
//    StopWordsRemover = new PredefinedStopWordsRemover(),
//    VectorNormalizer = TextTransformTextNormKind.L2,
//    CharFeatureExtractor = new NGramNgramExtractor() { NgramLength = 3, AllLengths = false },
//    WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = true }
//});

