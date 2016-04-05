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
        NegativeOpenChange = OpenPrevCloseDif * 2,
        NegativeCloseChange = NegativeOpenChange * 2,
        NegativeVolumeChange = NegativeCloseChange * 2,
        NegativeCloseOpenDif = NegativeVolumeChange * 2,
        NegativeOpenPrevCloseDif = NegativeCloseOpenDif * 2,
    }

    public enum TestDataAction
    {
        None,
        RemoveTestData,
        LeaveOnlyTestData,
    }

    #endregion
}
