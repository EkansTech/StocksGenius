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
        HighOpenDif = OpenPrevCloseDif * 2,
        HighCloseDif = HighOpenDif * 2,
        NegativeOpenChange = HighCloseDif * 2,
        NegativeCloseChange = NegativeOpenChange * 2,
        NegativeVolumeChange = NegativeCloseChange * 2,
        NegativeCloseOpenDif = NegativeVolumeChange * 2,
        NegativeOpenPrevCloseDif = NegativeCloseOpenDif * 2,
        NegativeLowOpenDif = NegativeOpenPrevCloseDif * 2,
        NegativeLowCloseDif = NegativeLowOpenDif * 2,
    }

    public enum TestDataAction
    {
        None,
        LoadOnlyPredictionData,
        LoadOnlyTestData,
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
