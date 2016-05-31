using StocksData;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class StockSimulation
    {
        #region Members

        private static DateTime m_SimulationDate;

        private double m_MaxTotalValue = 0.0;

        private double m_MinTotalValue = 0.0;

        private double m_MaxTotalValueLoose = 0.0;

        InvestmentAnalyzis m_InvestmentAnalyzis;

        private int m_SimulationRun = 0;

        public int m_TotalNumOfInvestments = 0;

        public int m_NumOfGoodInvestments = 0;

        public List<DataSet> m_TradableDataSets = new List<DataSet>();

        private Dictionary<Investment, int> m_SafeModeInvestments = new Dictionary<Investment, int>();

        private DataSetsMetaData m_MetaData;

        private Dictionary<Investment, DateTime> m_TodayReleased = new Dictionary<Investment, DateTime>();

        public List<double> m_TotalValues = new List<double>();

        // private Dictionary<DataSet, Dictionary<ulong, int>> m_GoodCombinations = new Dictionary<DataSet, Dictionary<ulong, int>>();

        // private Dictionary<DataSet, Dictionary<ulong, int>> m_BadCombinations = new Dictionary<DataSet, Dictionary<ulong, int>>();

        #endregion

        #region Properties

        public Dictionary<string, DataSet> DataSets { get; set; }

        public Dictionary<string, DataSet> PriceDataSets { get; set; }

        public Dictionary<string, double> StocksTotalProfit { get; set; }

        internal List<Investment> Investments { get; set; }

        static public byte MinDaysOfUp { get; set; }

        static public byte MaxPredictedRange { get; set; }

        static public double MinChangeForDown { get; set; }

        static public double MinProfitRatio { get; set; }

        static public int MaxInvestmentsPerStock { get; set; }

        static public double MaxLooseRatio { get; set; }

        static public string WorkingDirectory { get; set; }

        public static double PredictionErrorRange { get; set; }

        public static int MinDaysOfDown { get; set; }

        public static int MaxDaysUntilProfit { get; set; }

        public static int SafeDaysNum { get; set; }

        public static int MaxNumOfInvestments { get; set; }

        static public int MaxInvestmentsLive = 1;

        private double m_RealMoney = 2500;

        public double RealMoney
        {
            get { return m_RealMoney; }
            set { m_RealMoney = value; }
        }

        public double InvestmentsValue
        {
            get
            {
                return Investments.Sum(x => x.GetInvestmentValue(m_SimulationDate));
            }
        }
        public double TotalValue { get { return m_RealMoney + InvestmentsValue; } }

        public int NumOfInvestments { get { return Investments.Count(); } }

        public static DateTime StartDate { get; set; }

        public static int MonthsJump { get; set; }

        public static bool InvertInvestments = false;

        #endregion

        #region Constructors

        public StockSimulation(DataSetsMetaData metaData, string workingDirectory)
        {
            WorkingDirectory = workingDirectory;
            MinChangeForDown = DSSettings.EffectivePredictionResult;
            DataSets = new Dictionary<string, DataSet>();
            PriceDataSets = new Dictionary<string, DataSet>();
            StocksTotalProfit = new Dictionary<string, double>();
            Investments = new List<Investment>();
            m_MetaData = metaData;
            StartDate = SimSettings.SimulateSince;
            MonthsJump = SimSettings.SimulateEveryX;

            foreach (string dataSetCode in m_MetaData.Keys)
            {
                DataSet dataSet = new DataSet(dataSetCode, m_MetaData[dataSetCode].DataSetFilePath);
                DataSets.Add(dataSet.DataSetCode, dataSet);

                DataSet priceDataSet = new DataSet(dataSetCode, m_MetaData[dataSetCode].PriceDataSetFilePath);
                PriceDataSets.Add(priceDataSet.DataSetCode, priceDataSet);
                if (!StocksTotalProfit.ContainsKey(dataSetCode))
                {
                    StocksTotalProfit.Add(dataSet.DataSetCode, 0.0);
                }
            }
        }

        #endregion

        #region Interface

        public void Simulate()
        {
            m_MaxTotalValue = 0.0;
            m_MinTotalValue = 0.0;
            Log.AddMessage("Simulating, Investment money: {0}", m_RealMoney);

            //StartDate = new DateTime(2013, 1, 1);

            //   while (StartDate < SimSettings.SimulateUpTo)
            //  {
            MinChangeForDown = DSSettings.EffectivePredictionResult;
            //ReloadPredictions(StartDate);

            for (int minDaysOfDown = 1; minDaysOfDown <= 10; minDaysOfDown += 2)
            {
                MinDaysOfDown = minDaysOfDown;
                for (int maxDaysUntilProfit = 1; maxDaysUntilProfit <= 10; maxDaysUntilProfit += 2)
                {
                    MaxDaysUntilProfit = maxDaysUntilProfit;
                    for (double predictionErrorRange = 0.01; predictionErrorRange <= 0.01; predictionErrorRange += 0.1)
                    {
                        PredictionErrorRange = predictionErrorRange;
                        for (double minChangeForDown = 0.01; minChangeForDown <= 0.01; minChangeForDown += 0.01)
                        {
                            MinChangeForDown = minChangeForDown;
                            for (byte minDaysOfUp = 10; minDaysOfUp <= 20; minDaysOfUp += 2)
                            {
                                MinDaysOfUp = minDaysOfUp;
                                for (byte maxPredictedRange = SimSettings.MaxPredictedRange; maxPredictedRange <= SimSettings.MaxPredictedRange; maxPredictedRange += 1)
                                {
                                    MaxPredictedRange = maxPredictedRange;
                                    for (double minProfitRatio = 0.00; minProfitRatio <= 0.06; minProfitRatio += 0.02)
                                    {
                                        MinProfitRatio = minProfitRatio;
                                        for (double maxLooseRatio = 0; maxLooseRatio >= 0; maxLooseRatio -= 0.1)
                                        {
                                            MaxLooseRatio = maxLooseRatio;
                                            for (int maxInvestmentPerStock = 1; maxInvestmentPerStock <= 1; maxInvestmentPerStock++)
                                            {
                                                for (int safeDaysNum = 5; safeDaysNum <= 5; safeDaysNum++)
                                                {
                                                    for (int maxNumOfInvestments = 100; maxNumOfInvestments <= 100; maxNumOfInvestments++)
                                                    {
                                                        if (SimSettings.SimulatePerStock)
                                                        {
                                                            Dictionary<string, DataSet> tempDataSets = DataSets;
                                                            Dictionary<string, DataSet> tempPriceDataSets = PriceDataSets;
                                                            DataSets = new Dictionary<string, DataSet>();
                                                            PriceDataSets = new Dictionary<string, DataSet>();
                                                            foreach (string dataSetCode in tempDataSets.Keys)
                                                            {
                                                                DataSets.Add(dataSetCode, tempDataSets[dataSetCode]);
                                                                PriceDataSets.Add(dataSetCode, tempPriceDataSets[dataSetCode]);
                                                                RunSimulation();
                                                                DataSets.Clear();
                                                                PriceDataSets.Clear();
                                                            }
                                                        }
                                                        else
                                                        {
                                                            RunSimulation();
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Log.AddMessage("Final ammount of money: {0}", RealMoney);
            Log.AddMessage("Max total profit = {0}, min total profit = {1}", m_MaxTotalValue.ToString("0.00"), m_MinTotalValue.ToString("0.00"));
        }

        private void RunSimulation()
        {
            Log.IsActive = false;
            InvertInvestments = false;
            MaxNumOfInvestments = MaxNumOfInvestments;
            SafeDaysNum = SafeDaysNum;
            MaxInvestmentsPerStock = MaxInvestmentsPerStock;
            m_MaxTotalValue = SimSettings.RealMoneyStartValue;
            m_MinTotalValue = SimSettings.RealMoneyStartValue;
            m_MaxTotalValueLoose = 0.0;
            Investment.Reset();
            m_RealMoney = SimSettings.RealMoneyStartValue;
            m_TotalNumOfInvestments = 0;
            m_NumOfGoodInvestments = 0;
            m_SafeModeInvestments.Clear();
            m_TotalValues.Clear();

            foreach (string dataSetName in DataSets.Keys)
            {
                StocksTotalProfit[dataSetName] = 0.0;
            }

            m_InvestmentAnalyzis = new InvestmentAnalyzis(WorkingDirectory, m_SimulationRun);

            StockRecorder stockRecorder = new StockRecorder(StartDate, MinChangeForDown, MinProfitRatio, MaxInvestmentsPerStock, MaxLooseRatio, MinDaysOfUp,
                MaxPredictedRange, m_SimulationRun, PredictionErrorRange, MinDaysOfDown, MaxDaysUntilProfit, SafeDaysNum, MaxNumOfInvestments);

            //ReloadPredictions(StartDate);
            for (m_SimulationDate = StartDate; m_SimulationDate <= SimSettings.SimulateUpTo; m_SimulationDate = m_SimulationDate.AddDays(1))
            {
                if (m_SimulationDate == SimSettings.SimulateUpTo)
                {
                    List<Investment> investmentsToRelease = Investments.ToList();
                    foreach (Investment investment in investmentsToRelease)
                    {
                        investment.Action = ActionType.Released;
                        investment.ActionReason = ActionReason.EndOfTrade;
                        ReleaseInvestment(m_SimulationDate, investment);
                    }
                }
                else
                {
                    Log.AddMessage("Trade date: {0}", m_SimulationDate.ToShortDateString());
                    RunSimulationCycle(m_SimulationDate);
                    Log.AddMessage("Num of investments: {0}, Real Money: {1}, Investments value: {2}, Total value {3}", NumOfInvestments, m_RealMoney, InvestmentsValue, TotalValue);
                    stockRecorder.AddRecord(m_SimulationDate, m_RealMoney, TotalValue, NumOfInvestments);
                }
            }
            Log.AddMessage("Num of investments: {0}, Total value {1}", m_TotalNumOfInvestments, TotalValue);

            stockRecorder.SaveToFile(WorkingDirectory, m_MaxTotalValue, m_MinTotalValue, m_TotalNumOfInvestments, m_MaxTotalValueLoose, m_NumOfGoodInvestments);
            m_InvestmentAnalyzis.SaveToFile();
            m_SimulationRun++;
        }

        #endregion

        #region Private Methods

        private void RunSimulationCycle(DateTime day)
        {
            GetTradableDataSets(day);
            if (m_TradableDataSets.Count == 0)
            {
                return;
            }
            Log.AddMessage("{0}:", m_SimulationDate.ToShortDateString());

            m_TotalValues.Add(TotalValue);

            UpdateInvestments(day);

            ReleaseInvestments(day);

            CreateNewInvestments(day);

            m_TradableDataSets.Clear();

            List<Investment> keys = m_TodayReleased.Keys.ToList();
            foreach (Investment investment in keys)
            {
                if (investment.DataSet.GetDayNum(m_TodayReleased[investment]) - investment.DataSet.GetDayNum(day) >= investment.PredictedChange.Range)
                {
                    m_TodayReleased.Remove(investment);
                }
            }
        }

        private void GetTradableDataSets(DateTime day)
        {
            foreach (DataSet dataSet in DataSets.Values)
            {
                if (dataSet.ContainsTradeDay(day))
                {
                    m_TradableDataSets.Add(dataSet);
                }
            }
        }

        private void UpdateInvestments(DateTime day)
        {
            foreach (Investment investment in Investments.Where(x => m_TradableDataSets.Contains(x.DataSet)).OrderBy(x => x.ID))
            {
                investment.UpdateInvestment(day, TotalValue, RealMoney, StocksTotalProfit[investment.DataSet.DataSetCode]);
                m_InvestmentAnalyzis.Add(investment, day);
            }
        }

        private void ReleaseInvestments(DateTime day)
        {
            List<Investment> investmentsToRelease = Investments.Where(x => x.IsEndOfInvestment).Where(x => m_TradableDataSets.Contains(x.DataSet)).ToList();
            foreach (Investment investment in investmentsToRelease.OrderBy(x => x.ID))
            {
                ReleaseInvestment(day, investment);
                if (investment.ActionReason == ActionReason.GoodProfit || investment.ActionReason == ActionReason.BadLoose)
                {
                    m_TodayReleased.Add(investment, day);
                }
            }
        }

        private void ReleaseInvestment(DateTime day, Investment investment)
        {
            investment.UpdateRealMoneyOnRelease(day, ref m_RealMoney);
            if (TotalValue > m_MaxTotalValue)
            {
                m_MaxTotalValue = TotalValue;
            }
            else if (TotalValue < m_MinTotalValue)
            {
                m_MinTotalValue = TotalValue;
            }
            if (m_MaxTotalValue - TotalValue > m_MaxTotalValueLoose)
            {
                m_MaxTotalValueLoose = m_MaxTotalValue - TotalValue;
            }

            if (investment.Profit > 0)
            {
                m_NumOfGoodInvestments++;
            }

            StocksTotalProfit[investment.DataSet.DataSetCode] = investment.Release(day, TotalValue, StocksTotalProfit[investment.DataSet.DataSetCode]);
            Investments.Remove(investment);

            Log.AddMessage("Release investment of {0} with prediction {1}:", investment.DataSet.DataSetCode, investment.PredictedChange.ToString());
            Log.AddMessage("Release profit {0}, total value {1}, correctness {2}, {3} predictions", investment.GetProfit(day).ToString("0.00"),
                TotalValue.ToString("0.00"), investment.Analyze.AverageCorrectness.ToString("0.00"), investment.Analyze.NumOfPredictions);
        }

        private void AddInvestment(DateTime day, Analyze analyze, double addPercentagePrice)
        {
            if (analyze.PredictedChange.Range > analyze.DataSet.GetDayNum(day))
            {
                return;
            }
            Investment investment = new Investment(DataSets[analyze.DataSetName], PriceDataSets[analyze.DataSetName], analyze, day, TotalValue, RealMoney, StocksTotalProfit[analyze.DataSet.DataSetCode], addPercentagePrice);
            investment.UpdateAccountOnInvestment(day, TotalValue);
            investment.UpdateRealMoneyOnInvestment(day, ref m_RealMoney);
            Investments.Add(investment);

            if (TotalValue > m_MaxTotalValue)
            {
                m_MaxTotalValue = TotalValue;
            }
            else if (TotalValue < m_MinTotalValue)
            {
                m_MinTotalValue = TotalValue;
            }
            if (m_MaxTotalValue - TotalValue > m_MaxTotalValueLoose)
            {
                m_MaxTotalValueLoose = m_MaxTotalValue - TotalValue;
            }

            Log.AddMessage("New investment of {0} with prediction {1}, num of investments {2}:", investment.DataSet.DataSetCode, investment.PredictedChange.ToString(), NumOfInvestments);
            Log.AddMessage("Total Value {0}, {1} {2} shares, price {3}", TotalValue, (investment.InvestmentType == BuySell.Buy) ? "bought" : "sold", investment.Ammount, investment.InvestedPrice);

            m_InvestmentAnalyzis.Add(investment, day);
            m_TotalNumOfInvestments++;
        }

        private void CreateNewInvestments(DateTime day)
        {
            if (DataSets.Values.First().GetDayNum(day) == 0)
            {
                return;
            }

            foreach (DataSet dataSet in m_TradableDataSets)
            {
                if (dataSet.GetDate(0) == day)
                {
                    continue;
                }
                //if (Investments.Where(x => x.DataSet == dataSet).Count() >= MaxInvestmentsPerStock)
                //{
                //    continue;
                //}

                int numOfDowns = 0;
                for (int i = 0; i < 100; i++)
                {
                    int dayNum = dataSet.GetDayNum(day);
                    if (dataSet.NumOfRows > dayNum + i + 1 && (dataSet.GetData(dayNum + i, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + i + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + i + 1, DataSet.DataColumns.Open) < MinChangeForDown)
                    {
                        numOfDowns++;
                    }
                    else
                    {
                        break;
                    }
                }

                if (numOfDowns >= MinDaysOfDown)
                {
                    Analyze analyze = new Analyze()
                    {
                        AverageCorrectness = 1,
                        DataSet = dataSet,
                        DataSetName = dataSet.DataSetCode,
                        NumOfPredictions = 1,
                        PredictedChange = new CombinationItem(1, DataItem.OpenUp, 0, 0)
                    };
                    AddInvestment(day, analyze, 0);
                }

                int numOfUps = 0;
                for (int i = 0; i < 100; i++)
                {
                    int dayNum = dataSet.GetDayNum(day);
                    if (dataSet.NumOfRows > dayNum + i + 1 && (dataSet.GetData(dayNum + i, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + i + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + i + 1, DataSet.DataColumns.Open) > -MinChangeForDown)
                    {
                        numOfUps++;
                    }
                    else
                    {
                        break;
                    }
                }

                if (numOfUps >= MinDaysOfUp)
                {
                    Analyze analyze = new Analyze()
                    {
                        AverageCorrectness = 1,
                        DataSet = dataSet,
                        DataSetName = dataSet.DataSetCode,
                        NumOfPredictions = 1,
                        PredictedChange = new CombinationItem(1, DataItem.OpenDown, 0, 0)
                    };
                    AddInvestment(day, analyze, 0);
                }
            }            
        }

        private bool IsGoodInvestment(DataSet dataSet, BuySell investmentType, DateTime day)
        {
            double todayPrice = dataSet.GetDayData(day)[(int)DataSet.DataColumns.Open];
            double prevDayPrice = dataSet.GetDayData(day.AddDays(-1))[(int)DataSet.DataColumns.Open];

            return (investmentType == BuySell.Buy) ? todayPrice < prevDayPrice : todayPrice > prevDayPrice;
        }

        private double CalculateAverage(DataSet dataSet, int dataRow, int range, DataSet.DataColumns dataColum)
        {
            double sum = 0;
            for (int i = dataRow; i < dataRow + range; i++)
            {
                sum += dataSet[i * (int)DataSet.DataColumns.NumOfColumns + (int)dataColum];
            }

            return sum / range;
        }



        #endregion
    }
}
