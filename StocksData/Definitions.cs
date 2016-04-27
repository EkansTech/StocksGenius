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
        OpenChange = 1,
        CloseChange = OpenChange * 2,
        VolumeChange = CloseChange * 2,
        CloseOpenDif = VolumeChange * 2,
        OpenPrevCloseDif = CloseOpenDif * 2,
        PrevCloseOpenDif = OpenPrevCloseDif * 2,
        PrevHighOpenDif = PrevCloseOpenDif * 2,
        PrevHighCloseDif = PrevHighOpenDif * 2,
        NegativeOpenChange = PrevHighCloseDif * 2,
        NegativeCloseChange = NegativeOpenChange * 2,
        NegativeVolumeChange = NegativeCloseChange * 2,
        NegativeCloseOpenDif = NegativeVolumeChange * 2,
        NegativeOpenPrevCloseDif = NegativeCloseOpenDif * 2,
        NegativePrevCloseOpenDif = NegativeOpenPrevCloseDif * 2,
        NegativePrevLowOpenDif = NegativePrevCloseOpenDif * 2,
        NegativePrevLowCloseDif = NegativePrevLowOpenDif * 2,
    }

    public enum TestDataAction
    {
        None,
        LoadOnlyPredictionData,
        LoadOnlyTestData,
    }

    public struct ChangeMap
    {
        public const int NumOfData = 5;
        public DataSet.DataColumns FromData;
        public DataSet.DataColumns OfData;
        public int FromOffset;
        public int OfOffset;
        public bool IsPositiveChange;

        public ChangeMap(DataSet.DataColumns fromData, DataSet.DataColumns ofData, int fromOffset, int ofOffset, bool isPositiveChange)
        {
            FromData = fromData;
            OfData = ofData;
            FromOffset = fromOffset;
            OfOffset = ofOffset;
            IsPositiveChange = isPositiveChange;
        }
    }

    #endregion
}
