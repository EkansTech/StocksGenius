using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public static class SGSettings
    {
        public static double EffectivePredictionResult = 0.95;

        public static double PredictionErrorRange = 0.01;

        public static double MinProfitRatio = 0.3;

        public static double MaxLooseRatio = 0.3;

        public static double BuySellPenalty = 0.003;

        public static double SafesForStockRate = 0.05;

        public static double InvestmentPerStock = 5000;

        public static string DataSetsCodesPrefix = "WIKI-";

        public static DateTime PredictionsSince = new DateTime(2010, 1, 1);

        public static int PredictionEveryXMonths = 3;

        public static DataSourceTypes DataSourceType = DataSourceTypes.Yahoo;

        public const string RootDiretory = @"C:\Ekans\Stocks\";

        public static string Workspace = @"C:\Ekans\Stocks\iForex\";

        public const string InvestmentsFileName = "Investments.csv";

        public const string InvestmentsFile = "Investments.csv";

        public const string StockSettingsIni = "StocksSettings.ini";

        public const string WorkspaceSettingsIni = "WorkspaceSettings.ini";

        public const string InvestorIni = "Investor.ini";

        public const string NewOpenDataFile = "TodayOpenData.csv";
    }
}
