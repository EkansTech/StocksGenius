﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
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

        LimitLooseReopen,
        

        SamePredictions,

        EndOfPeriod,
        PredictionInverse,
        GoodProfit,
        BadLoose,
        EndOfTrade,
        MaxInvestmentLive,
        SafeMode,
        ProfitLimit,
        LooseLimit,
    }

}
