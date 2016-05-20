﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class SimSettings
    {
        public const double BuySellPenalty = 0.002;

        public const double InvestmentPerStock = 5000;

        public const string SimulationRecordsDirectory = "\\SimulationRecords\\";

        public const double MaxProfit = 1000;

        public const double MaxLoose = -1000;

        public static double SafesForStockRate = 0.05;

        public static double RealMoneyStartValue = 2500;
    };
}
    