using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public static class DSSettings
    {
        public static readonly List<CombinationItem> PredictionItems = new List<CombinationItem>()
        {
            new CombinationItem(3, DataItem.OpenChange),
            new CombinationItem(3, DataItem.NegativeOpenChange),
            new CombinationItem(5, DataItem.OpenChange),
            new CombinationItem(5, DataItem.NegativeOpenChange),
            new CombinationItem(7, DataItem.OpenChange),
            new CombinationItem(7, DataItem.NegativeOpenChange),
            new CombinationItem(9, DataItem.OpenChange),
            new CombinationItem(9, DataItem.NegativeOpenChange),
            new CombinationItem(11, DataItem.OpenChange),
            new CombinationItem(11, DataItem.NegativeOpenChange),
        };

        public static readonly List<CombinationItem> ChangeItems = new List<CombinationItem>()
        {
            new CombinationItem(3, DataItem.OpenChange),
            new CombinationItem(3, DataItem.NegativeOpenChange),
            new CombinationItem(3, DataItem.VolumeChange),
            new CombinationItem(3, DataItem.NegativeVolumeChange),
            new CombinationItem(5, DataItem.OpenChange),
            new CombinationItem(5, DataItem.NegativeOpenChange),
            new CombinationItem(5, DataItem.VolumeChange),
            new CombinationItem(5, DataItem.NegativeVolumeChange),
            new CombinationItem(8, DataItem.OpenChange),
            new CombinationItem(8, DataItem.NegativeOpenChange),
            new CombinationItem(8, DataItem.VolumeChange),
            new CombinationItem(8, DataItem.NegativeVolumeChange),
            new CombinationItem(12, DataItem.OpenChange),
            new CombinationItem(12, DataItem.NegativeOpenChange),
            new CombinationItem(12, DataItem.VolumeChange),
            new CombinationItem(12, DataItem.NegativeVolumeChange),
            new CombinationItem(16, DataItem.OpenChange),
            new CombinationItem(16, DataItem.NegativeOpenChange),
            new CombinationItem(16, DataItem.VolumeChange),
            new CombinationItem(16, DataItem.NegativeVolumeChange),
            new CombinationItem(20, DataItem.OpenChange),
            new CombinationItem(20, DataItem.NegativeOpenChange),
            new CombinationItem(20, DataItem.VolumeChange),
            new CombinationItem(20, DataItem.NegativeVolumeChange),
        };

        public const string DataSetsDir = "\\DataSets\\";

        public const string PredictionSuffix = "-Predictions";

        public const string PredictionDir = "\\Predictions\\";

        public const string LatestPredictionsSuffix = "-LatestPredictions";
                            
        public const string LatestPredictionsDir = "\\LatestPredictions\\";

        public const string DataSetCodesFile = "WIKI-datasets-codes.csv";

        public const int RelevantHistory = 100;

        public const double PredictionErrorRange = 0.000;

        public const int MinimumChangesForPrediction = 100;

        public const int TestRange = 100;

        public const int PredictionsSize = 300;

        public readonly static int DataSetForPredictionsSize = PredictionsSize + 3 * ChangeItems.OrderByDescending(x => x.Range).First().Range;

        public readonly static int TestMinSize = TestRange + 2 * ChangeItems.OrderByDescending(x => x.Range).First().Range;

        public const int PredictionMaxCombinationSize = 24;

        public const double MinimumRelevantPredictionResult = 0.8;

        public static readonly int GPUCycleSize = 1024 * 1024;

        public const double GPUHopelessPredictionLimit = 0.0;

        public readonly static List<DataItem> DataItems = DataPredictions.GetDataItems();

        public static readonly Dictionary<CombinationItem, byte> ChangeItemsMap = ChangeItems.Select(x => (byte)ChangeItems.IndexOf(x)).ToDictionary(x => ChangeItems[x]);
        public static readonly Dictionary<ulong, CombinationItem> ULongToCombinationItemMap = ChangeItems.ToDictionary(x => x.ToULong());

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
    }
}
