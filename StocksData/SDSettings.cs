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

        public const string PredictionDataSetSuffix = "-Predictions";

        public const string PredictionDataSetsDir = "\\Predictions\\";

        public const string DataSetCodesFile = "WIKI-datasets-codes.csv";

        public const int RelevantHistory = 100;

        public const float PredictionErrorRange = 0.000F;

        public const int MinimumChangesForPrediction = 50;

        public const int TestRange = 100;

        public static readonly List<CombinationItem> PredictionItems = new List<CombinationItem>()
        {
            new CombinationItem(3, DataItem.OpenChange),
            new CombinationItem(3, DataItem.NegativeOpenChange),
            new CombinationItem(6, DataItem.OpenChange),
            new CombinationItem(6, DataItem.NegativeOpenChange),
            new CombinationItem(9, DataItem.OpenChange),
            new CombinationItem(9, DataItem.NegativeOpenChange),
        };

        public const int PredictionMaxCombinationSize = 9;

        public const float MinimumRelevantPredictionResult = 0.85F;

        public static readonly int GPUCycleSize = 1024 * 1024;

        public readonly static List<DataItem> DataItems = DataPredictions.GetDataItems();

        public static readonly List<CombinationItem> ChangeItems = new List<CombinationItem>()
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

        public const float GPUHopelessPredictionLimit = 0.0F;
    }
}
