using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public static class SGSettings
    {
        public const double InvestmentMoney = 1000;

        public const string WorkingDirectory = @"C:\Ekans\Stocks\iForex\";

        public const string InvestmentsFile = "Investments.csv";

        public const double EffectivePredictionResult = 0.95;

        public static double PredictionErrorRange = 0.01;

        public static double MinProfitRatio = 0.3;

        public static double MaxLooseRatio = 0.3;

        public static double BuySellPenalty = 0.001;

        public const string InvestmentsFileName = "Investments.csv";

        public const double SafesForStockRate = 0.05;

        public const double InvestmentPerStock = 5000;

        public const string InvestorIni = "Investor.ini";
    }
}
