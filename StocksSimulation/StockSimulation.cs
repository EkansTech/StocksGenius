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

        private double m_MaxTotalProfit = 0.0;

        private double m_MinTotalProfit = 0.0;

        private double m_MaxTotalProfitLoose = 0.0;

        InvestmentAnalyzis m_InvestmentAnalyzis;

        private int m_SimulationRun = 0;

        public int m_TotalNumOfInvestments = 0;

        public int m_NumOfGoodInvestments = 0;

        public List<DataSet> m_TradableDataSets = new List<DataSet>();

        private Dictionary<Investment, int> m_SafeModeInvestments = new Dictionary<Investment, int>();

        private DataSetsMetaData m_MetaData;

        private Dictionary<Investment, DateTime> m_TodayReleased = new Dictionary<Investment, DateTime>();

        public List<double> m_TotalProfits = new List<double>();

        // private Dictionary<DataSet, Dictionary<ulong, int>> m_GoodCombinations = new Dictionary<DataSet, Dictionary<ulong, int>>();

        // private Dictionary<DataSet, Dictionary<ulong, int>> m_BadCombinations = new Dictionary<DataSet, Dictionary<ulong, int>>();

        private Dictionary<DateTime, Dictionary<double, List<Analyze>>> m_DailyAnalyzes = new Dictionary<DateTime, Dictionary<double, List<Analyze>>>();

        private List<Analyze> m_PotentialAnalyzes = null;

        private List<Investment> m_TodayCreated = new List<Investment>();

        #endregion

        #region Properties

        public Dictionary<string, DataSet> DataSets { get; set; }

        public Dictionary<string, double> StocksTotalProfit { get; set; }

        internal List<Investment> Investments { get; set; }

        static public byte MinDaysOfUp { get; set; }

        static public double MinLimitLooseReopen { get; set; }

        static public double MinChangeForDown { get; set; }

        static public double MinProfitRatio { get; set; }

        static public int MaxInvestmentsPerStock { get; set; }

        static public double MaxLooseLimitRelease { get; set; }

        static public string WorkingDirectory { get; set; }

        public static double MinChangeForUp { get; set; }

        public static int MinDaysOfDown { get; set; }

        public static int MaxDaysUntilProfit { get; set; }

        public static int NumOfDayWithoutProift { get; set; }

        public static int MaxNumOfInvestments { get; set; }

        public static double ChangeErrorRange { get; set; }

        static public int MaxInvestmentsLive = 1;

        private double m_RealMoney = 2500;

        private int m_LastMaxDay;

        private int m_TradableDayNum;

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

        private double m_TotalProfit;

        public double TotalProfit
        {
            get { return m_TotalProfit; }
            set { m_TotalProfit = value; }
        }


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
            StocksTotalProfit = new Dictionary<string, double>();
            Investments = new List<Investment>();
            m_MetaData = metaData;
            StartDate = SimSettings.SimulateSince;
            MonthsJump = SimSettings.SimulateEveryX;

            foreach (string dataSetCode in m_MetaData.Keys)
            {
                DataSet dataSet = new DataSet(dataSetCode, m_MetaData[dataSetCode].DataSetFilePath);
                DataSets.Add(dataSet.DataSetCode, dataSet);

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
            m_MaxTotalProfit = 0.0;
            m_MinTotalProfit = 0.0;
            Log.AddMessage("Simulating, Investment money: {0}", m_RealMoney);

            //StartDate = new DateTime(2013, 1, 1);

            //   while (StartDate < SimSettings.SimulateUpTo)
            //  {
            MinChangeForDown = DSSettings.EffectivePredictionResult;
            //ReloadPredictions(StartDate);

            for (int minDaysOfDown = 40; minDaysOfDown <= 40; minDaysOfDown += 3)
            {
                MinDaysOfDown = minDaysOfDown;
                for (double minChangeForDown = -0.1; minChangeForDown >= -0.2; minChangeForDown -= 0.01)
                {
                    MinChangeForDown = minChangeForDown;
                    for (byte minDaysOfUp = 0; minDaysOfUp <= 0; minDaysOfUp += 3)
                    {
                        MinDaysOfUp = minDaysOfUp;
                        for (double minChangeForUp = 0; minChangeForUp <= 0; minChangeForUp += 0.06)
                        {
                            MinChangeForUp = minChangeForUp;
                            for (int maxDaysUntilProfit = 1; maxDaysUntilProfit <= 9; maxDaysUntilProfit += 2)
                            {
                                MaxDaysUntilProfit = maxDaysUntilProfit;
                                for (double minProfitRatio = 1; minProfitRatio <= 1; minProfitRatio += 1)
                                {
                                    MinProfitRatio = minProfitRatio;
                                    for (double minLimitLooseReopen = 0; minLimitLooseReopen <= 0; minLimitLooseReopen += 0.01)
                                    {
                                        MinLimitLooseReopen = minLimitLooseReopen;
                                        for (double maxLooseLimitRelease = -0; maxLooseLimitRelease >= -0; maxLooseLimitRelease -= 0.01)
                                        {
                                            MaxLooseLimitRelease = maxLooseLimitRelease;
                                            for (int maxInvestmentPerStock = 1; maxInvestmentPerStock <= 1; maxInvestmentPerStock++)
                                            {
                                                MaxInvestmentsPerStock = maxInvestmentPerStock;
                                                for (double changeErrorRange = -0.02; changeErrorRange <= 0.005; changeErrorRange += 0.005)
                                                {
                                                    ChangeErrorRange = changeErrorRange;
                                                    for (int maxNumOfInvestments = 1000; maxNumOfInvestments <= 1000; maxNumOfInvestments++)
                                                    {
                                                        MaxNumOfInvestments = maxNumOfInvestments;
                                                        if (SimSettings.SimulatePerStock)
                                                        {
                                                            Dictionary<string, DataSet> tempDataSets = DataSets;
                                                            DataSets = new Dictionary<string, DataSet>();
                                                            foreach (string dataSetCode in tempDataSets.Keys)
                                                            {
                                                                DataSets.Add(dataSetCode, tempDataSets[dataSetCode]);
                                                                RunSimulation();
                                                                DataSets.Clear();
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
            Log.AddMessage("Max total profit = {0}, min total profit = {1}", m_MaxTotalProfit.ToString("0.00"), m_MinTotalProfit.ToString("0.00"));
        }

        private void RunSimulation()
        {
            Log.IsActive = false;
            InvertInvestments = false;
            m_MaxTotalProfit = SimSettings.RealMoneyStartValue;
            m_MinTotalProfit = SimSettings.RealMoneyStartValue;
            m_MaxTotalProfitLoose = 0.0;
            Investment.Reset();
            m_RealMoney = SimSettings.RealMoneyStartValue;
            m_TotalNumOfInvestments = 0;
            m_NumOfGoodInvestments = 0;
            m_SafeModeInvestments.Clear();
            m_TotalProfits.Clear();
            m_LastMaxDay = 0;
            m_TradableDayNum = 0;
            NumOfDayWithoutProift = 0;
            m_TotalProfit = 0;

            foreach (string dataSetName in DataSets.Keys)
            {
                StocksTotalProfit[dataSetName] = 0.0;
            }

            m_InvestmentAnalyzis = new InvestmentAnalyzis(WorkingDirectory, m_SimulationRun);

            StockRecorder stockRecorder = new StockRecorder(StartDate, MinChangeForDown, MinProfitRatio, MaxInvestmentsPerStock, MaxLooseLimitRelease, MinDaysOfUp,
                MinLimitLooseReopen, m_SimulationRun, MinChangeForUp, MinDaysOfDown, MaxDaysUntilProfit, MaxNumOfInvestments, ChangeErrorRange);

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
                        m_InvestmentAnalyzis.Add(investment, m_SimulationDate);
                    }
                }
                else
                {
                    Log.AddMessage("Trade date: {0}", m_SimulationDate.ToShortDateString());
                    RunSimulationCycle(m_SimulationDate);
                    Log.AddMessage("Num of investments: {0}, Real Money: {1}, Investments value: {2}, Total value {3}", NumOfInvestments, m_RealMoney, InvestmentsValue, TotalValue);
                    stockRecorder.AddRecord(m_SimulationDate, m_RealMoney, TotalValue, m_TotalProfit, NumOfInvestments);
                }
            }
            Console.WriteLine("Sim {0}: Num of investments: {1}, Total value {2}, ProfitPerInvestment {3}, No profit days {4}",m_SimulationRun,  m_TotalNumOfInvestments, TotalValue, TotalValue / m_TotalNumOfInvestments, NumOfDayWithoutProift);

            stockRecorder.SaveToFile(WorkingDirectory, m_MaxTotalProfit, m_MinTotalProfit, m_TotalNumOfInvestments, m_MaxTotalProfitLoose, m_NumOfGoodInvestments, NumOfDayWithoutProift);
            m_InvestmentAnalyzis.SaveToFile();
            m_SimulationRun++;
        }

        #endregion

        #region Private Methods

        private void RunSimulationCycle(DateTime day)
        {
            m_TodayCreated.Clear();
            m_TodayReleased.Clear();
            GetTradableDataSets(day);
            if (m_TradableDataSets.Count == 0)
            {
                return;
            }
            Log.AddMessage("{0}:", m_SimulationDate.ToShortDateString());
            m_TradableDayNum++;

            GetAnalyzes(day);

            UpdateInvestments(day);

            ReleaseInvestments(day);

            CreateNewInvestments(day);

            ReleaseIfProfitLimit(day);

            //ReleaseIfLooseLimit(day);

            m_TradableDataSets.Clear();

            List<Investment> keys = m_TodayReleased.Keys.ToList();
            foreach (Investment investment in keys)
            {
                if (investment.DataSet.GetDayNum(m_TodayReleased[investment]) - investment.DataSet.GetDayNum(day) >= investment.PredictedChange.Range)
                {
                    m_TodayReleased.Remove(investment);
                }
            }

            m_TotalProfits.Add(TotalProfit);

            if (m_TotalProfits.Last() > m_TotalProfits[m_LastMaxDay])
            {
                m_LastMaxDay = m_TradableDayNum;
            }
            else if (m_TradableDayNum - m_LastMaxDay > NumOfDayWithoutProift)
            {
                NumOfDayWithoutProift = m_TradableDayNum - m_LastMaxDay;
            }
        }

        private void GetAnalyzes(DateTime day)
        {
            if (m_DailyAnalyzes.ContainsKey(day) && m_DailyAnalyzes[day].ContainsKey(ChangeErrorRange))
            {
                m_PotentialAnalyzes = m_DailyAnalyzes[day][ChangeErrorRange].Where(x => (x.Change < 0) ?
                x.Change < MinChangeForDown || x.SequenceLength >= MinDaysOfDown
                :
                x.Change > MinChangeForUp || x.SequenceLength >= MinDaysOfUp).ToList();
                //m_PotentialAnalyzes = m_DailyAnalyzes[day][ChangeErrorRange].Where(x => (x.Change < 0) ?
                //x.Change < MinChangeForDown && x.SequenceLength >= MinDaysOfDown
                //:
                //x.Change > MinChangeForUp && x.SequenceLength >= MinDaysOfUp).ToList();
                //Best 44 days
                return;
            }

            m_PotentialAnalyzes = new List<Analyze>();

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

                for (int numOfDowns = 0; numOfDowns < 100; numOfDowns++)
                {
                    int dayNum = dataSet.GetDayNum(day);
                    if (dataSet.NumOfRows > dayNum + numOfDowns + 1
                        && ((dataSet.GetData(dayNum + numOfDowns, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfDowns + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfDowns + 1, DataSet.DataColumns.Open) < ChangeErrorRange)
                      && ((dataSet.GetData(dayNum + numOfDowns + 1, DataSet.DataColumns.High) - dataSet.GetData(dayNum + numOfDowns + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfDowns + 1, DataSet.DataColumns.Open) < 0.02))
                    {
                    }
                    else
                    {
                        Analyze analyze = new Analyze()
                        {
                            Change = (dataSet.GetData(dayNum, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfDowns, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfDowns, DataSet.DataColumns.Open),
                            LastChange = numOfDowns == 0 ? 0 : (dataSet.GetData(dayNum, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + 1, DataSet.DataColumns.Open),
                            DataSet = dataSet,
                            DataSetName = dataSet.DataSetCode,
                            SequenceLength = numOfDowns,
                            PredictedChange = new CombinationItem(1, DataItem.OpenUp, 0, 0)
                        };
                        analyze.MinProfitRatio = Math.Max(-analyze.Change / (double)analyze.SequenceLength, -analyze.LastChange) * MinProfitRatio * 2;
                        //AddInvestment(day, analyze, 0);
                        m_PotentialAnalyzes.Add(analyze);
                        break;
                    }
                }

                for (int numOfUps = 0; numOfUps < 100; numOfUps++)
                {
                    int dayNum = dataSet.GetDayNum(day);
                    if (dataSet.NumOfRows > dayNum + numOfUps + 1
                        && ((dataSet.GetData(dayNum + numOfUps, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfUps + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfUps + 1, DataSet.DataColumns.Open) > -ChangeErrorRange)
                        && ((dataSet.GetData(dayNum + numOfUps + 1, DataSet.DataColumns.Low) - dataSet.GetData(dayNum + numOfUps + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfUps + 1, DataSet.DataColumns.Open) > -0.01))
                    {
                    }
                    else
                    {
                        Analyze analyze = new Analyze()
                        {
                            Change = (dataSet.GetData(dayNum, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfUps, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfUps, DataSet.DataColumns.Open),
                            LastChange = numOfUps == 0 ? 0 : (dataSet.GetData(dayNum, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + 1, DataSet.DataColumns.Open),
                            DataSet = dataSet,
                            DataSetName = dataSet.DataSetCode,
                            SequenceLength = numOfUps,
                            PredictedChange = new CombinationItem(1, DataItem.OpenDown, 0, 0)
                        };
                        analyze.MinProfitRatio = Math.Max(analyze.Change / (double)analyze.SequenceLength, analyze.LastChange) * MinProfitRatio / 2;
                        //AddInvestment(day, analyze, 0);
                        //m_PotentialAnalyzes.Add(analyze);
                        break;
                    }
                }

            }

            m_PotentialAnalyzes = m_PotentialAnalyzes.Where(x => (x.Change < 0) ?
            x.Change < -0.05 || x.SequenceLength >= 5
            :
            x.Change > 0.05 || x.SequenceLength >= 5).ToList();

            //m_PotentialAnalyzes = m_PotentialAnalyzes.OrderByDescending(x => (x.Change < 0 ? -x.Change * 100 + 10 : x.Change * 100) + x.SequenceLength).ToList();
            m_PotentialAnalyzes = m_PotentialAnalyzes.OrderByDescending(x => (x.LastChange < 0 ? -x.LastChange * 100 + 10 : x.LastChange * 100) + x.SequenceLength).ToList();

            if (!m_DailyAnalyzes.ContainsKey(day))
            {
                m_DailyAnalyzes.Add(day, new Dictionary<double, List<Analyze>>());
            }

            m_DailyAnalyzes[day].Add(ChangeErrorRange, m_PotentialAnalyzes);

            m_PotentialAnalyzes = m_PotentialAnalyzes.Where(x => (x.Change < 0) ?
            x.Change < MinChangeForDown || x.SequenceLength >= MinDaysOfDown
            :
            x.Change > MinChangeForUp || x.SequenceLength >= MinDaysOfUp).ToList();
            //m_PotentialAnalyzes = m_PotentialAnalyzes.Where(x => (x.Change < 0) ?
            //x.Change < MinChangeForDown && x.SequenceLength >= MinDaysOfDown
            //:
            //x.Change > MinChangeForUp && x.SequenceLength >= MinDaysOfUp).ToList();
        }

        private void ReleaseIfLooseLimit(DateTime day)
        {
            List<Investment> tradableInvestments = Investments.Where(x => m_TradableDataSets.Contains(x.DataSet)).ToList();
            double releasedLoose = 0.0;
            List<Investment> keys = m_TodayCreated.Where(x => Investments.Contains(x)).ToList();
            foreach (Investment investment in keys)
            {
                if (investment.GetMaxDailyLoosePercentage(day) < MaxLooseLimitRelease)
                {
                    investment.Action = ActionType.Released;
                    investment.ActionReason = ActionReason.LooseLimit;
                    if (investment.GetOpenLoosePercentage(day) < MaxLooseLimitRelease)
                    {
                        releasedLoose = investment.GetOpenLoosePercentage(day);
                    }
                    else
                    {
                        releasedLoose = MaxLooseLimitRelease;
                    }
                    ReleaseInvestment(day, investment, true, releasedLoose);
                    m_TodayReleased.Add(investment, day);
                    m_InvestmentAnalyzis.Add(investment, day);
                }

                if (investment.GetMaxDailyLoosePercentage(day) < releasedLoose - MinLimitLooseReopen)
                {
                    AddInvestment(day, investment.Analyze, investment.GetPriceFromRatio(day, releasedLoose - MinLimitLooseReopen), ActionReason.LimitLooseReopen);
                }
            }
        }

        private void ReleaseIfProfitLimit(DateTime day)
        {
            List<Investment> tradableInvestments = Investments.Where(x => m_TradableDataSets.Contains(x.DataSet)).ToList();
            foreach (Investment investment in tradableInvestments)
            {
                if (investment.GetMaxDailyProfitPercentage(day) > investment.Analyze.MinProfitRatio)
                {
                    investment.Action = ActionType.Released;
                    investment.ActionReason = ActionReason.ProfitLimit;

                    ReleaseInvestment(day, investment, true, investment.Analyze.MinProfitRatio);
                    m_TodayReleased.Add(investment, day);
                    m_InvestmentAnalyzis.Add(investment, day);
                }
            }
        }

        private void GetTradableDataSets(DateTime day)
        {
            foreach (DataSet dataSet in DataSets.Values)
            {
                if (dataSet.IsTradableDay(day))
                {
                    m_TradableDataSets.Add(dataSet);
                }
            }
        }

        private void UpdateInvestments(DateTime day)
        {
            foreach (Investment investment in Investments.Where(x => m_TradableDataSets.Contains(x.DataSet)).OrderBy(x => x.ID))
            {
                investment.UpdateInvestment(day, TotalValue, m_TotalProfit, RealMoney, StocksTotalProfit[investment.DataSet.DataSetCode]);
                if (investment.Action != ActionType.Released)
                {
                    m_InvestmentAnalyzis.Add(investment, day);
                }
            }
        }

        private void ReleaseInvestments(DateTime day)
        {
            List<Investment> investmentsToRelease = Investments.Where(x => x.IsEndOfInvestment).Where(x => m_TradableDataSets.Contains(x.DataSet)).ToList();
            foreach (Investment investment in investmentsToRelease.OrderBy(x => x.ID))
            {
                Analyze investmentAnalyze = m_PotentialAnalyzes.FirstOrDefault(x => x.DataSet == investment.DataSet);
                if (investmentAnalyze != null)// && m_PotentialAnalyzes.IndexOf(investmentAnalyze) < MaxNumOfInvestments)
                {
                    investment.Action = ActionType.Continued;
                    investment.ActionReason = ActionReason.SamePredictions;
                    m_InvestmentAnalyzis.Add(investment, day);
                    continue;
                }
                ReleaseInvestment(day, investment);
                if (investment.ActionReason == ActionReason.GoodProfit || investment.ActionReason == ActionReason.BadLoose)
                {
                    m_TodayReleased.Add(investment, day);
                }

                m_InvestmentAnalyzis.Add(investment, day);
            }
        }

        private void ReleaseInvestment(DateTime day, Investment investment, bool limitRelease = false, double profit = 0.0)
        {
            if (!limitRelease)
            {
                investment.UpdateRealMoneyOnRelease(day, ref m_RealMoney);
                StocksTotalProfit[investment.DataSet.DataSetCode] = investment.Release(day, TotalValue, m_TotalProfit, StocksTotalProfit[investment.DataSet.DataSetCode]);
                m_TotalProfit += investment.Profit;
            }
            else
            {
                StocksTotalProfit[investment.DataSet.DataSetCode] = investment.LimitRelease(day, TotalValue, m_TotalProfit, StocksTotalProfit[investment.DataSet.DataSetCode], profit);
                m_TotalProfit += investment.Profit;
                investment.UpdateRealMoneyOnRelease(day, ref m_RealMoney);
            }

            if (TotalProfit > m_MaxTotalProfit)
            {
                m_MaxTotalProfit = TotalProfit;
            }
            else if (TotalProfit < m_MinTotalProfit)
            {
                m_MinTotalProfit = TotalProfit;
            }
            if (m_MaxTotalProfit - TotalProfit > m_MaxTotalProfitLoose)
            {
                m_MaxTotalProfitLoose = m_MaxTotalProfit - TotalProfit;
            }

            if (investment.Profit > 0)
            {
                m_NumOfGoodInvestments++;
            }

            Investments.Remove(investment);

            Log.AddMessage("Release investment of {0} with prediction {1}:", investment.DataSet.DataSetCode, investment.PredictedChange.ToString());
            Log.AddMessage("Release profit {0}, total value {1}, change {2}, {3} predictions", investment.GetProfit(day).ToString("0.00"),
                TotalValue.ToString("0.00"), investment.Analyze.Change.ToString("0.00"), investment.Analyze.SequenceLength);
        }

        private void AddInvestment(DateTime day, Analyze analyze, double addPercentagePrice, ActionReason reason)
        {
            if (analyze.PredictedChange.Range > analyze.DataSet.GetDayNum(day))
            {
                return;
            }
            Investment investment = new Investment(DataSets[analyze.DataSetName], analyze, day, TotalValue, m_TotalProfit, RealMoney, StocksTotalProfit[analyze.DataSet.DataSetCode], addPercentagePrice, reason);
            investment.UpdateAccountOnInvestment(day, TotalValue);
            investment.UpdateRealMoneyOnInvestment(day, ref m_RealMoney);
            Investments.Add(investment);

            if (TotalProfit > m_MaxTotalProfit)
            {
                m_MaxTotalProfit = TotalProfit;
            }
            else if (TotalProfit < m_MinTotalProfit)
            {
                m_MinTotalProfit = TotalProfit;
            }
            if (m_MaxTotalProfit - TotalProfit > m_MaxTotalProfitLoose)
            {
                m_MaxTotalProfitLoose = m_MaxTotalProfit - TotalProfit;
            }

            Log.AddMessage("New investment of {0} with prediction {1}, num of investments {2}:", investment.DataSet.DataSetCode, investment.PredictedChange.ToString(), NumOfInvestments);
            Log.AddMessage("Total Value {0}, {1} {2} shares, price {3}", TotalValue, (investment.InvestmentType == BuySell.Buy) ? "Bought" : "Sold", investment.Ammount, investment.InvestedPrice);

            m_InvestmentAnalyzis.Add(investment, day);
            m_TodayCreated.Add(investment);
            m_TotalNumOfInvestments++;
        }

        private void CreateNewInvestments(DateTime day)
        {
            if (DataSets.Values.First().GetDayNum(day) == 0)
            {
                return;
            }

            Analyze secondChance;
            //if (Investments.Count > 0 && (secondChance = potentialAnalyzes.FirstOrDefault(x => x.DataSet == Investments.First().DataSet)) != null)
            //{
            //    AddInvestment(day, secondChance, 0);
            //    return;
            //}
            //else

            int analyzeNum = 0;
            while (Investments.Count < MaxNumOfInvestments && analyzeNum < m_PotentialAnalyzes.Count)
            {
                AddInvestment(day, m_PotentialAnalyzes[analyzeNum], 0, ActionReason.NoReason);
                analyzeNum++;
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
