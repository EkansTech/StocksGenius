using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public static class DSSettings
    {
        public static List<CombinationItem> PredictionItems = new List<CombinationItem>();

        public static readonly List<CombinationItem> ChangeItems = new List<CombinationItem>();

        public const string DataSetsDir = "DataSets\\";

        public const string PriceDataSetsDir = "DataSets\\";

        public const string PredictionSuffix = "-Predictions";

        public const string PredictionDir = "Predictions\\";

        public const string SimPredictionDir = "SimPredictions\\";

        public const string LatestPredictionsSuffix = "-LatestPredictions";
                            
        public const string LatestPredictionsDir = "LatestPredictions\\";

        public const string DataSetsMetaDataFile = "datasets-metadata.csv";

        public const string CombinedDataPredictionsFile = "CombinedDataPredictions.csv";

        public static double PredictionErrorRange = 0.01;

        public static double MinimumChangesForPredictionRatio = 0.01;

        public static double EffectivePredictionResult = 0.9;

        public static int PredictionMaxCombinationSize = 7;

        public static DateTime DataRelevantSince = new DateTime(2014, 1, 1);

        public const int TestRange = 100;

        public const int PredictionsSize = 300;

        public static int MaxChangeRange { get { return ChangeItems.Select(x => x.Range).Max(); } }

        public static int MaxPredictionRange { get { return PredictionItems.Select(x => x.Range).Max(); } }

        public static int DataSetForPredictionsSize { get { return PredictionsSize + 3 * MaxChangeRange; } }

        public static int TestMinSize { get { return TestRange + 3 * MaxChangeRange; } }

        public static readonly int GPUCycleSize = 1024 * 1024;

        public const double GPUHopelessPredictionLimit = 0.0;

        public readonly static List<DataItem> DataItems = DataPredictions.GetDataItems();

        private static Dictionary<CombinationItem, byte> m_ChangeItemsMap = null;

        public static Dictionary<CombinationItem, byte> ChangeItemsMap
        {
            get { return (m_ChangeItemsMap == null) ? m_ChangeItemsMap = ChangeItems.Select(x => (byte)ChangeItems.IndexOf(x)).ToDictionary(x => ChangeItems[x]) : m_ChangeItemsMap; }
        }

        private static Dictionary<ulong, CombinationItem> m_ULongToCombinationItemMap = null;

        public static Dictionary<ulong, CombinationItem> ULongToCombinationItemMap
        {
            get { return (m_ULongToCombinationItemMap == null) ? m_ULongToCombinationItemMap = ChangeItems.ToDictionary(x => x.ToULong()) : m_ULongToCombinationItemMap; }
        }

        public static readonly List<DataItem> PositiveChanges = new List<DataItem>()
        {
            DataItem.OpenUp,
            DataItem.CloseUp,
            DataItem.VolumeUp,
            DataItem.CloseOpenPositive,
            DataItem.OpenPrevClosePositive,
            DataItem.HighOpenPositive,
            DataItem.HighClosePositive,
        };

        public static readonly List<DataItem> NegativeChanges = new List<DataItem>()
        {
            DataItem.OpenDown,
            DataItem.CloseDown,
            DataItem.VolumeDown,
            DataItem.CloseOpenNegative,
            DataItem.OpenPrevCloseNegative,
            DataItem.LowOpenNegative,
            DataItem.LowCloseNegative,
        };

        public static readonly Dictionary<DataItem, DataItem> OppositeDataItems = new Dictionary<DataItem, DataItem>()
        {
            { DataItem.OpenUp, DataItem.OpenDown },
            { DataItem.CloseUp, DataItem.CloseDown },
            { DataItem.VolumeUp, DataItem.VolumeDown },
            { DataItem.CloseOpenPositive, DataItem.CloseOpenNegative },
            { DataItem.OpenPrevClosePositive, DataItem.OpenPrevCloseNegative },
            { DataItem.HighOpenPositive, DataItem.None },
            { DataItem.HighClosePositive, DataItem.None },
            { DataItem.OpenDown, DataItem.OpenUp },
            { DataItem.CloseDown, DataItem.CloseUp },
            { DataItem.VolumeDown, DataItem.VolumeUp },
            { DataItem.CloseOpenNegative, DataItem.CloseOpenPositive },
            { DataItem.OpenPrevCloseNegative, DataItem.OpenPrevClosePositive },
            { DataItem.LowOpenNegative, DataItem.None },
            { DataItem.LowCloseNegative, DataItem.None },
        };

        public static readonly Dictionary<DataItem, ChangeMap> DataItemsCalculationMap = new Dictionary<DataItem, ChangeMap>()
        {
            { DataItem.OpenUp, new ChangeMap(DataSet.DataColumns.Open, DataSet.DataColumns.Open, 0, 1, true, 0)  },
            { DataItem.CloseUp, new ChangeMap(DataSet.DataColumns.Close, DataSet.DataColumns.Close, 0, 1, true, 1)  },
            { DataItem.VolumeUp, new ChangeMap(DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, 0, 1, true, 1)  },
            { DataItem.CloseOpenPositive, new ChangeMap(DataSet.DataColumns.Close, DataSet.DataColumns.Open, 0, 0, true, 1)  },
            { DataItem.OpenPrevClosePositive, new ChangeMap(DataSet.DataColumns.Open, DataSet.DataColumns.Close, 0, 1, true, 0)  },
            { DataItem.HighOpenPositive, new ChangeMap(DataSet.DataColumns.High, DataSet.DataColumns.Open, 0, 0, true, 1)  },
            { DataItem.HighClosePositive, new ChangeMap(DataSet.DataColumns.High, DataSet.DataColumns.Close, 0, 0, true, 1)  },
            { DataItem.OpenDown, new ChangeMap(DataSet.DataColumns.Open, DataSet.DataColumns.Open, 0, 1, false, 0)  },
            { DataItem.CloseDown, new ChangeMap(DataSet.DataColumns.Close, DataSet.DataColumns.Close, 0, 1, false, 1)  },
            { DataItem.VolumeDown, new ChangeMap(DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, 0, 1, false, 1)  },
            { DataItem.CloseOpenNegative, new ChangeMap(DataSet.DataColumns.Close, DataSet.DataColumns.Open, 0, 0, false, 1)  },
            { DataItem.OpenPrevCloseNegative, new ChangeMap(DataSet.DataColumns.Open, DataSet.DataColumns.Close, 0, 1, false, 0)  },
            { DataItem.LowOpenNegative, new ChangeMap(DataSet.DataColumns.Low, DataSet.DataColumns.Open, 0, 0, false, 1)  },
            { DataItem.LowCloseNegative, new ChangeMap(DataSet.DataColumns.Low, DataSet.DataColumns.Close, 0, 0, false, 1)  },
        };
    }
}
