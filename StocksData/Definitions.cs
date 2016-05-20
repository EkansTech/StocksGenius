using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    #region Enums

    [Flags]
    public enum DataItem
    {
        None = 0,
        OpenUp = 1,
        CloseUp = OpenUp * 2,
        VolumeUp = CloseUp * 2,
        CloseOpenPositive = VolumeUp * 2,
        OpenPrevClosePositive = CloseOpenPositive * 2,
        HighOpenPositive = OpenPrevClosePositive * 2,
        HighClosePositive = HighOpenPositive * 2,
        OpenDown = HighClosePositive * 2,
        CloseDown = OpenDown * 2,
        VolumeDown = CloseDown * 2,
        CloseOpenNegative = VolumeDown * 2,
        OpenPrevCloseNegative = CloseOpenNegative * 2,
        LowOpenNegative = OpenPrevCloseNegative * 2,
        LowCloseNegative = LowOpenNegative * 2,
    }

    public enum TestDataAction
    {
        None,
        LoadLimitedPredictionData,
        LoadOnlyTestData,
        LoadWithoutTestData,
    }

    public enum DataSourceTypes
    {
        Quandl,
        Yahoo,
        Xignite,
        Bloomberg,
    }

    public struct ChangeMap
    {
        public const int NumOfData = 6;
        public DataSet.DataColumns FromData;
        public DataSet.DataColumns OfData;
        public int FromOffset;
        public int OfOffset;
        public bool IsPositiveChange;
        public int Offset;

        public ChangeMap(DataSet.DataColumns fromData, DataSet.DataColumns ofData, int fromOffset, int ofOffset, bool isPositiveChange, int offset)
        {
            FromData = fromData;
            OfData = ofData;
            FromOffset = fromOffset;
            OfOffset = ofOffset;
            IsPositiveChange = isPositiveChange;
            Offset = offset;
        }
    }

    #endregion
}
