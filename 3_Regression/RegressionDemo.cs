using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace BeerML.Regression
{
    public class PriceData
    {
        [Column(ordinal: "0")]
        public string FullName;
        [Column(ordinal: "1", name: "Label")]
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
            // Define pipeline
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader("3_Regression/problem3_train.csv").CreateFrom<PriceData>(useHeader: true, separator: ','));

            pipeline.Add(new ColumnConcatenator("NumericalFeatures",
                "Volume"));

            pipeline.Add(new ColumnConcatenator("CategoryFeatures",
                 "Country", "Type"));

            pipeline.Add(new TextFeaturizer("FullName", "FullName"));

            pipeline.Add(new CategoricalOneHotVectorizer("CategoryFeatures"));
            pipeline.Add(new ColumnConcatenator("Features",
                "FullName", "NumericalFeatures", "CategoryFeatures"));

            //pipeline.Add(new StochasticDualCoordinateAscentRegressor());
            pipeline.Add(new PoissonRegressor());

            // Train model
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var model = pipeline.Train<PriceData, PricePrediction>();
            stopWatch.Stop();
            Console.WriteLine($"Trained the model in: {stopWatch.ElapsedMilliseconds / 1000} seconds.");

            // Evaluate model
            //var testData = new TextLoader<PriceData>("3_Regression/problem3_validate.csv", useHeader: true, separator: ",");

            //var evaluator = new RegressionEvaluator();
            //RegressionMetrics metrics = evaluator.Evaluate(model, testData);

            // Use model
            IEnumerable<PriceData> drinks = new[]
            {
                new PriceData { FullName="Cheap Lager", Type="Öl", Volume=500, Country="Sverige" },
                new PriceData { FullName="Dummy Weiss", Type="Öl", Volume=500, Country="Tyskland" },
                new PriceData { FullName="New Trappist", Type="Öl", Volume=330, Country="Belgien" },
                new PriceData { FullName="Mortgage 10 years", Type="Whisky", Volume=700, Country="Storbritannien" },
                new PriceData { FullName="Mortgage 21 years", Type="Whisky", Volume=700, Country="Storbritannien" },
                new PriceData { FullName="Merlot Classic", Type="Rött vin", Volume=750, Country="Frankrike" },
                new PriceData { FullName="Merlot Grand Cru", Type="Rött vin", Volume=750, Country="Frankrike" },
                new PriceData { FullName="Château de la Victoria Cru", Type="Rött vin", Volume=750, Country="Frankrike" }
            };

            var predictions = model.Predict(drinks).ToList();

        }

    }
}
