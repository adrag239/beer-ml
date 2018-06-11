using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;

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
            // Define pipeline
            //var pipeline = new LearningPipeline();

            //pipeline.Add(new TextLoader("4_Clustering/problem4.csv").CreateFrom<ClusteringData>(useHeader: true, separator: ','));

            //pipeline.Add(new TextFeaturizer("Features", "FullName")
            //{
            //    WordFeatureExtractor = new NGramNgramExtractor() { NgramLength = 2, AllLengths = false },
            //    KeepNumbers = true,
            //    Language = TextTransformLanguage.English
            //});

            //pipeline.Add(new KMeansPlusPlusClusterer() { K = 2 });

            // Train model
            //var stopWatch = new Stopwatch();
            //stopWatch.Start();
            //var model = pipeline.Train<ClusteringData, ClusteringPrediction>();
            //model.WriteAsync("problem4.model.zip");
            //stopWatch.Stop();
            //Console.WriteLine($"Trained the model in: {stopWatch.ElapsedMilliseconds / 1000} seconds.");

            var model = PredictionModel.ReadAsync<ClusteringData, ClusteringPrediction>("4_Clustering/problem4.model.zip").GetAwaiter().GetResult();

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

            var predictions = model.Predict(drinks).ToList();

        }
    }
}
