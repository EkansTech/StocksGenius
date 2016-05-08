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
            new CombinationItem(1, DataItem.OpenChange),
            new CombinationItem(1, DataItem.NegativeOpenChange),
            new CombinationItem(2, DataItem.OpenChange),
            new CombinationItem(2, DataItem.NegativeOpenChange),
            new CombinationItem(3, DataItem.OpenChange),
            new CombinationItem(3, DataItem.NegativeOpenChange),
            new CombinationItem(4, DataItem.OpenChange),
            new CombinationItem(4, DataItem.NegativeOpenChange),
            new CombinationItem(5, DataItem.OpenChange),
            new CombinationItem(5, DataItem.NegativeOpenChange),
            new CombinationItem(6, DataItem.OpenChange),
            new CombinationItem(6, DataItem.NegativeOpenChange),
            //new CombinationItem(7, DataItem.OpenChange),
            //new CombinationItem(7, DataItem.NegativeOpenChange),
            new CombinationItem(8, DataItem.OpenChange),
            new CombinationItem(8, DataItem.NegativeOpenChange),
            //new CombinationItem(9, DataItem.OpenChange),
            //new CombinationItem(9, DataItem.NegativeOpenChange),
            new CombinationItem(10, DataItem.OpenChange),
            new CombinationItem(10, DataItem.NegativeOpenChange),
            //new CombinationItem(11, DataItem.OpenChange),
            //new CombinationItem(11, DataItem.NegativeOpenChange),
            new CombinationItem(12, DataItem.OpenChange),
            new CombinationItem(12, DataItem.NegativeOpenChange),
        };

        public static readonly List<CombinationItem> ChangeItems = new List<CombinationItem>()
        {
            new CombinationItem(1, DataItem.CloseOpenDif),
            new CombinationItem(1, DataItem.NegativeCloseOpenDif),
            new CombinationItem(1, DataItem.OpenPrevCloseDif),
            new CombinationItem(1, DataItem.NegativeOpenPrevCloseDif),
            new CombinationItem(1, DataItem.OpenChange),
            new CombinationItem(1, DataItem.NegativeOpenChange),
            //new CombinationItem(1, DataItem.CloseOpenDif),
            //new CombinationItem(1, DataItem.NegativeCloseOpenDif),
            new CombinationItem(1, DataItem.VolumeChange),
            new CombinationItem(1, DataItem.NegativeVolumeChange),
            new CombinationItem(1, DataItem.HighOpenDif),
            new CombinationItem(1, DataItem.NegativeLowOpenDif),
            new CombinationItem(1, DataItem.HighCloseDif),
            new CombinationItem(1, DataItem.NegativeLowCloseDif),
            new CombinationItem(2, DataItem.OpenChange),
            new CombinationItem(2, DataItem.NegativeOpenChange),
            new CombinationItem(2, DataItem.VolumeChange),
            new CombinationItem(2, DataItem.NegativeVolumeChange),
            new CombinationItem(3, DataItem.OpenChange),
            new CombinationItem(3, DataItem.NegativeOpenChange),
            new CombinationItem(3, DataItem.VolumeChange),
            new CombinationItem(3, DataItem.NegativeVolumeChange),
            //new CombinationItem(4, DataItem.OpenChange),
            //new CombinationItem(4, DataItem.NegativeOpenChange),
            //new CombinationItem(4, DataItem.VolumeChange),
            //new CombinationItem(4, DataItem.NegativeVolumeChange),
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
            new CombinationItem(21, DataItem.OpenChange),
            new CombinationItem(21, DataItem.NegativeOpenChange),
            new CombinationItem(21, DataItem.VolumeChange),
            new CombinationItem(21, DataItem.NegativeVolumeChange),
        };

        public const string DataSetsDir = "\\DataSets\\";

        public const string PriceDataSetsDirectory = "\\PriceDataSets\\";

        public const string PredictionSuffix = "-Predictions";

        public const string PredictionDir = "\\Predictions\\";

        public const string LatestPredictionsSuffix = "-LatestPredictions";
                            
        public const string LatestPredictionsDir = "\\LatestPredictions\\";

        public const string DataSetCodesFile = "datasets-codes.csv";

        public const double PredictionErrorRange = 0.01;

        public const double MinimumChangesForPredictionRatio = 0.01;

        public const int TestRange = 100;

        public const int PredictionsSize = 300;

        public readonly static int MaxChangeRange = ChangeItems.Select(x => x.Range).Max();

        public readonly static int MaxPredictionRange = PredictionItems.Select(x => x.Range).Max();

        public readonly static int DataSetForPredictionsSize = PredictionsSize + 3 * MaxChangeRange;

        public readonly static int TestMinSize = TestRange + 3 * MaxChangeRange;

        public const int PredictionMaxCombinationSize = 9;

        public const double MinimumRelevantPredictionResult = 0.9;

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
            DataItem.HighOpenDif,
            DataItem.HighCloseDif,
        };

        public static readonly List<DataItem> NegativeChanges = new List<DataItem>()
        {
            DataItem.NegativeOpenChange,
            DataItem.NegativeCloseChange,
            DataItem.NegativeVolumeChange,
            DataItem.NegativeCloseOpenDif,
            DataItem.NegativeOpenPrevCloseDif,
            DataItem.NegativeLowOpenDif,
            DataItem.NegativeLowCloseDif,
        };

        public static readonly Dictionary<DataItem, DataItem> OppositeDataItems = new Dictionary<DataItem, DataItem>()
        {
            { DataItem.OpenChange, DataItem.NegativeOpenChange },
            { DataItem.CloseChange, DataItem.NegativeCloseChange },
            { DataItem.VolumeChange, DataItem.NegativeVolumeChange },
            { DataItem.CloseOpenDif, DataItem.NegativeCloseOpenDif },
            { DataItem.OpenPrevCloseDif, DataItem.NegativeOpenPrevCloseDif },
            { DataItem.HighOpenDif, DataItem.None },
            { DataItem.HighCloseDif, DataItem.None },
            { DataItem.NegativeOpenChange, DataItem.OpenChange },
            { DataItem.NegativeCloseChange, DataItem.CloseChange },
            { DataItem.NegativeVolumeChange, DataItem.VolumeChange },
            { DataItem.NegativeCloseOpenDif, DataItem.CloseOpenDif },
            { DataItem.NegativeOpenPrevCloseDif, DataItem.OpenPrevCloseDif },
            { DataItem.NegativeLowOpenDif, DataItem.None },
            { DataItem.NegativeLowCloseDif, DataItem.None },
        };

        public static readonly Dictionary<DataItem, ChangeMap> DataItemsCalculationMap = new Dictionary<DataItem, ChangeMap>()
        {
            { DataItem.OpenChange, new ChangeMap(DataSet.DataColumns.Open, DataSet.DataColumns.Open, 0, 1, true, 0)  },
            { DataItem.CloseChange, new ChangeMap(DataSet.DataColumns.Close, DataSet.DataColumns.Close, 0, 1, true, 1)  },
            { DataItem.VolumeChange, new ChangeMap(DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, 0, 1, true, 1)  },
            { DataItem.CloseOpenDif, new ChangeMap(DataSet.DataColumns.Close, DataSet.DataColumns.Open, 0, 0, true, 1)  },
            { DataItem.OpenPrevCloseDif, new ChangeMap(DataSet.DataColumns.Open, DataSet.DataColumns.Close, 0, 1, true, 0)  },
            { DataItem.HighOpenDif, new ChangeMap(DataSet.DataColumns.High, DataSet.DataColumns.Open, 0, 0, true, 1)  },
            { DataItem.HighCloseDif, new ChangeMap(DataSet.DataColumns.High, DataSet.DataColumns.Close, 0, 0, true, 1)  },
            { DataItem.NegativeOpenChange, new ChangeMap(DataSet.DataColumns.Open, DataSet.DataColumns.Open, 0, 1, false, 0)  },
            { DataItem.NegativeCloseChange, new ChangeMap(DataSet.DataColumns.Close, DataSet.DataColumns.Close, 0, 1, false, 1)  },
            { DataItem.NegativeVolumeChange, new ChangeMap(DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, 0, 1, false, 1)  },
            { DataItem.NegativeCloseOpenDif, new ChangeMap(DataSet.DataColumns.Close, DataSet.DataColumns.Open, 0, 0, false, 1)  },
            { DataItem.NegativeOpenPrevCloseDif, new ChangeMap(DataSet.DataColumns.Open, DataSet.DataColumns.Close, 0, 1, false, 0)  },
            { DataItem.NegativeLowOpenDif, new ChangeMap(DataSet.DataColumns.Low, DataSet.DataColumns.Open, 0, 0, false, 1)  },
            { DataItem.NegativeLowCloseDif, new ChangeMap(DataSet.DataColumns.Low, DataSet.DataColumns.Close, 0, 0, false, 1)  },
        };
    }
}
