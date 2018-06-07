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

namespace BeerMl
{
	class Program
	{
		static void Main(string[] args)
		{
            Problem1();

            //Problem2();

            //Problem3();
		}

		private static void Problem1()
		{
			// Define pipeline
			var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader("problem1_train.csv").CreateFrom<BeerOrWineData>(useHeader: true, separator: ','));

			pipeline.Add(new TextFeaturizer("Features", "FullName"));

			pipeline.Add(new Dictionarizer(("Type", "Label")));

			pipeline.Add(new StochasticDualCoordinateAscentBinaryClassifier() { });

			pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

			// Train model
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var model = pipeline.Train<BeerOrWineData, BeerOrWinePrediction>();
            stopWatch.Stop();
            Console.WriteLine($"Trained the model in: {stopWatch.ElapsedMilliseconds / 1000} seconds.");

			// Evaluate model
            var testData = new TextLoader("problem1_validate.csv").CreateFrom<BeerOrWineData>(useHeader: true, separator: ',');

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

			Console.ReadLine();
		}

		private static void Problem2()
		{
			// Define pipeline
			var pipeline = new LearningPipeline();

			pipeline.Add(new TextLoader("problem2_train.csv").CreateFrom<DrinkData>(useHeader: true, separator: ','));

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
            var testData = new TextLoader("problem2_validate.csv").CreateFrom<DrinkData>(useHeader: true, separator: ',');

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

			Console.ReadLine();
		}

		private static void Problem3()
		{
			// Define pipeline
			var pipeline = new LearningPipeline();

			pipeline.Add(new TextLoader("problem3_train.csv").CreateFrom<PriceData>(useHeader: true, separator: ','));

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
            Console.WriteLine($"Trained the model in: {stopWatch.ElapsedMilliseconds/1000} seconds.");

            //model.WriteAsync("problem3.model.zip");

            // Evaluate model
            //var testData = new TextLoader<PriceData>("problem3_validate.csv", useHeader: true, separator: ",");

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

			Console.ReadLine();
		}

	}


	#region "Problem 1"

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

	#endregion

	#region "Problem 2"

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

	#endregion

	#region "Problem 3"

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

    #endregion
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

//pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 25, NumTrees = 25, MinDocumentsInLeafs = 2 });   // up to 91% 
//pipeline.Add(new FastForestBinaryClassifier() { NumLeaves = 25, NumTrees = 25, MinDocumentsInLeafs = 2 });  // 86%
//pipeline.Add(new StochasticDualCoordinateAscentBinaryClassifier() { }); // 95%
//pipeline.Add(new StochasticGradientDescentBinaryClassifier { }); // 92%