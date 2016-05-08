using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public enum BuySell
    {
        None,
        Buy,
        Sell
    }

    public enum ActionType
    {
        NoAction,
        Created,
        Continued,
        Released,
    }

    public enum ActionReason
    {
        NoReason,

        SamePredictions,

        EndOfPeriod,
        PredictionInverse,
        GoodProfit,
        BadLoose,
        EndOfTrade,
        MaxInvestmentLive,
    }

    public enum InvestmentStatus
    {
        Active,
        Released,
    }
}
