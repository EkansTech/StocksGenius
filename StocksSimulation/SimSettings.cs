using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class SimSettings
    {
        public static double BuySellPenalty = 0.002;

        public const double InvestmentPerStock = 5000;

        public const string SimulationRecordsDirectory = "\\SimulationRecords\\";

        public const double MaxProfit = 1000;

        public const double MaxLoose = -1000;

        public static double SafesForStockRate = 0.05;

        public static double RealMoneyStartValue = 0;

        public static DateTime SimulateSince = new DateTime(2010, 1, 1);

        public static DateTime SimulateUpTo = DateTime.Today;

        public static int SimulateEveryX = 3;

        public static TimeType SimulateEveryXType = TimeType.Month;

        public static byte MinPredictedRange = 1;

        public static byte MaxPredictedRange = 2;

        public static bool TestAllRanges = false;

        public static bool SimulatePerStock = false;

        private static DateTime m_SimStartTime = DateTime.MinValue;

        public static DateTime SimStartTime
        {
            get
            {
                if (m_SimStartTime == DateTime.MinValue)
                {
                    m_SimStartTime = DateTime.Now;
                }

                return m_SimStartTime;
            }
        }

    };
}
    