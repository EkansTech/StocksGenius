using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public static class Constants
    {
        public const string DataSetsDir = "\\DataSets\\";

        public const string ChangeDataSetsDir = "\\ChangeDataSets\\";

        public const string PredictionsDataSetsDir = "\\PredictionDataSets\\";

        public const string AnalyzesDataSetsDir = "\\AnalyzeDataSets\\";

        public const string ChangeDataSetSuffix = "-Changes";

        public const string PredictionDataSetSuffix = "-Prediction";

        public const string AnalyzeDataSetSuffix = "-Analyze";

        public const int MinDepthRange = 10;

        public const int MaxDepthRange = 50;

        public const int DepthsNum = MaxDepthRange - MinDepthRange + 1;

        public const int NumOfOutOfRangePredictions = DepthsNum * (MaxDepthRange + MinDepthRange) / 2;

        public const int RelevantHistory = 100;

        public const int AnalyzeMaxCombinationSize = 5;

        public const double PredictionErrorRange = 0.001;

        public const int MinimumPredictionForAnalyze = 50;

        public static readonly List<ChangesDataSet.DataColumns> AnalyzeChangesList = new List<ChangesDataSet.DataColumns>()
        {
            ChangesDataSet.DataColumns.CloseOpenDif,
            ChangesDataSet.DataColumns.OpenPrevCloseDif,
        };
    }
}
