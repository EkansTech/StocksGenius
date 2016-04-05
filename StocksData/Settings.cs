using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public static class DSSettings
    {
        public const string DataSetsDir = "\\DataSets\\";

        public const string ChangeDataSetsDir = "\\ChangeDataSets\\";

        public const string PredictionsDataSetsDir = "\\PredictionDataSets\\";

        public const string AnalyzesDataSetsDir = "\\AnalyzeDataSets\\";

        public const string ChangeDataSetSuffix = "-Changes";

        public const string PredictionDataSetSuffix = "-Prediction";

        public const string AnalyzeDataSetSuffix = "-Analyze";

        public const int RelevantHistory = 100;

        public const float PredictionErrorRange = 0.000F;

        public const int MinimumPredictionsForAnalyze = 50;

        public const int TestRange = 100;

        public static readonly List<CombinationItem> AnalyzeItems = new List<CombinationItem>()
        {
            new CombinationItem(1, DataItem.CloseOpenDif),
            new CombinationItem(1, DataItem.OpenPrevCloseDif),
            new CombinationItem(1, DataItem.NegativeCloseOpenDif),
            new CombinationItem(1, DataItem.NegativeOpenPrevCloseDif),
            new CombinationItem(3, DataItem.OpenChange),
            new CombinationItem(3, DataItem.NegativeOpenChange),
            new CombinationItem(6, DataItem.OpenChange),
            new CombinationItem(6, DataItem.NegativeOpenChange),
            new CombinationItem(9, DataItem.OpenChange),
            new CombinationItem(9, DataItem.NegativeOpenChange),
        };

        public const int AnalyzeMaxCombinationSize = 6;

        public const string AnalyzerDataSetSuffix = "-Analyzer";

        public const string AnalyzerDataSetsDir = "\\Analyzer\\";

        public const float MinimumRelevantAnalyzeResult = 0.7F;

        public static readonly int GPUCycleSize = 1024 * 1024;// * AnalyzeItems.Count;

        public readonly static List<DataItem> DataItems = DataAnalyzer.GetDataItems();

        public static readonly List<CombinationItem> PredictionItems = new List<CombinationItem>()
        {
            new CombinationItem(1, DataItem.CloseOpenDif),
            new CombinationItem(1, DataItem.OpenPrevCloseDif),
            new CombinationItem(1, DataItem.NegativeCloseOpenDif),
            new CombinationItem(1, DataItem.NegativeOpenPrevCloseDif),
            new CombinationItem(1, DataItem.CloseChange),
            new CombinationItem(1, DataItem.NegativeCloseChange),
            new CombinationItem(1, DataItem.VolumeChange),
            new CombinationItem(1, DataItem.NegativeVolumeChange),
            new CombinationItem(3, DataItem.OpenChange),
            new CombinationItem(3, DataItem.NegativeOpenChange),
            new CombinationItem(3, DataItem.VolumeChange),
            new CombinationItem(3, DataItem.NegativeVolumeChange),
            new CombinationItem(6, DataItem.OpenChange),
            new CombinationItem(6, DataItem.NegativeOpenChange),
            new CombinationItem(6, DataItem.VolumeChange),
            new CombinationItem(6, DataItem.NegativeVolumeChange),
            new CombinationItem(9, DataItem.OpenChange),
            new CombinationItem(9, DataItem.NegativeOpenChange),
            new CombinationItem(9, DataItem.VolumeChange),
            new CombinationItem(9, DataItem.NegativeVolumeChange),
            new CombinationItem(12, DataItem.OpenChange),
            new CombinationItem(12, DataItem.NegativeOpenChange),
            new CombinationItem(12, DataItem.VolumeChange),
            new CombinationItem(12, DataItem.NegativeVolumeChange),
            new CombinationItem(15, DataItem.OpenChange),
            new CombinationItem(15, DataItem.NegativeOpenChange),
            new CombinationItem(15, DataItem.VolumeChange),
            new CombinationItem(15, DataItem.NegativeVolumeChange),
            new CombinationItem(18, DataItem.OpenChange),
            new CombinationItem(18, DataItem.NegativeOpenChange),
            new CombinationItem(18, DataItem.VolumeChange),
            new CombinationItem(18, DataItem.NegativeVolumeChange),

        };

        public static readonly Dictionary<CombinationItem, byte> PredictionItemsMap = PredictionItems.Select(x => (byte)PredictionItems.IndexOf(x)).ToDictionary(x => PredictionItems[x]);
        public static readonly Dictionary<ulong, CombinationItem> ULongToCombinationItemMap = PredictionItems.ToDictionary(x => x.ToULong());

        public static readonly List<DataItem> PositiveChanges = new List<DataItem>()
        {
            DataItem.OpenChange,
            DataItem.CloseChange,
            DataItem.VolumeChange,
            DataItem.CloseOpenDif,
            DataItem.OpenPrevCloseDif,
        };

        public static readonly List<DataItem> NegativeChanges = new List<DataItem>()
        {
            DataItem.NegativeOpenChange,
            DataItem.NegativeCloseChange,
            DataItem.NegativeVolumeChange,
            DataItem.NegativeCloseOpenDif,
            DataItem.NegativeOpenPrevCloseDif,
        };

        public const float GPUHopelessPredictionLimit = 0.0F;
    }
}
