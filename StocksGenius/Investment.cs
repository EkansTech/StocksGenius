using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using StocksData;

namespace StocksGenius
{
    public class Investment
    {
        #region Members

        private static int m_IDs = 0;

        private static int m_ReleaseIDs = 0;

        #endregion

        #region Properties

        public int ID { get; internal set; }

        public int ReleaseID { get; internal set; }

        public DataSet DataSet
        {
            get
            {
                return Investor.StocksData.DataSets[DataSetCode];
            }
        }

        public string DataSetCode { get; set; }

        public int Ammount { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public Analyze Analyze { get; set; }

        public DateTime InvestmentDay { get; set; }

        public DateTime ReleaseDay { get; set; }

        public double InvestedPrice { get; set; }

        public double ReleasePrice { get; set; }

        public double InvestedMoney { get; set; }

        public BuySell InvestmentType { get; set; }

        public ActionType Action { get; set; }

        public ActionReason ActionReason { get; set; }

        public double StockTotalProfit { get; set; }

        public InvestmentStatus Status { get; set; }

        #endregion

        #region Constructors

        public Investment(int id)
        {
            ID = id;
        }

        public Investment(Analyze analyze, DateTime day, double stockTotalProfit, double investingMoney, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            DataSetCode = analyze.DataSetCode;
            PredictedChange = analyze.PredictedChange;
            InvestmentDay = day;
            InvestedPrice = DataSet.GetDayData(day)[(int)dataColumn];
            Ammount = (int)(investingMoney / InvestedPrice);
            Analyze = analyze;
            InvestmentType = (analyze.IsPositiveInvestment) ? BuySell.Buy : BuySell.Sell; // Test opposite decision
            InvestedMoney = GetInvestmentMoney(InvestedPrice, InvestmentType);
            Action = ActionType.Created;
            ActionReason = ActionReason.NoReason;
            StockTotalProfit = stockTotalProfit;
            Status = InvestmentStatus.Active;
        }

        public Investment(string dataSetName, DateTime day, double stockTotalProfit, BuySell investmentType, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            DataSetCode = dataSetName;
            //PredictedChange = null;
            InvestmentDay = day;
            InvestedPrice = DataSet.GetDayData(day)[(int)dataColumn];
            Ammount = (int)(SGSettings.InvestmentPerStock / InvestedPrice);
            Analyze = null;
            InvestmentType = investmentType;
            InvestedMoney = GetInvestmentMoney(InvestedPrice, InvestmentType);
            Action = ActionType.Created;
            ActionReason = ActionReason.NoReason;
            StockTotalProfit = stockTotalProfit;
            Status = InvestmentStatus.Active;
        }

        private Investment(Investment investment)
        {
            ID = investment.ID;
            DataSetCode = investment.DataSetCode;
            PredictedChange = investment.PredictedChange;
            InvestmentDay = investment.InvestmentDay;
            InvestedPrice = investment.InvestedPrice;
            Ammount = investment.Ammount;
            Analyze = investment.Analyze;
            InvestmentType = investment.InvestmentType;
            InvestedMoney = investment.InvestedMoney;
            Action = investment.Action;
            ActionReason = investment.ActionReason;
            StockTotalProfit = investment.StockTotalProfit;
            ReleaseDay = investment.ReleaseDay;
            Status = InvestmentStatus.Active;
        }

        #endregion

        #region Interface

        public double GetInvestmentValue(DateTime day)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return SGSettings.SafesForStockRate * InvestedMoney + (GetReleaseMoney(day) - InvestedMoney);
            }
            else
            {
                return SGSettings.SafesForStockRate * InvestedMoney + (InvestedMoney - GetReleaseMoney(day));
            }
        }

        internal static void Reset(int id, int releaseId)
        {
            m_IDs = id;
            m_ReleaseIDs = releaseId;
        }

        internal double ProfitPercentage(DateTime day)
        {
            return GetProfit(day) / InvestedMoney * 100;
        }

        internal double GetDayPrice(DateTime day)
        {
            return DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open];
        }

        public Investment Clone()
        {
            return new Investment(this);
        }

        public static BuySell TimeToInvest(DataSet dataSet, List<Investment> investments, int day)
        {
            //if (dataSet.GetContinuousChange(day, DataSet.DataColumns.Open, 1) > StockSimulation.MinimumChange)
            //{
            //    return BuySell.Buy;
            //}
            //else if (dataSet.GetContinuousChange(day, DataSet.DataColumns.Open, 1) < -StockSimulation.MinimumChange)
            //{
            //    return BuySell.Sell;
            //}

            return BuySell.None;
        }
        
        public void UpdateInvestment(DailyAnalyzes dailyAnalyzes, DateTime day, double stockTotalProfit)
        {
            Action = ActionType.NoAction;
            StockTotalProfit = stockTotalProfit;

            //if (InvestmentDay - day >= AnalyzerSimulator.MaxInvestmentsLive)
            //{
            //    Action = Action.Released;
            //    ActionReason = ActionReason.MaxInvestmentLive;
            //    IsEndOfInvestment = true;
            //    return;
            //}

            //if (day == 0)
            //{
            //    Action = ActionType.Released;
            //    ActionReason = ActionReason.EndOfTrade;
            //    IsEndOfInvestment = true;
            //}

            int daysLeft = PredictedChange.Range - (DataSet.GetDayNum(InvestmentDay) - DataSet.GetDayNum(day));

            if (dailyAnalyzes.ContainsKey(DataSetCode))
            {
                var dataSetAnalyzes = dailyAnalyzes[DataSetCode].Values.OrderBy(x => x.PredictedChange.Range);
                //if ((InvestmentType == BuySell.Buy && dataSetAnalyzes.First().IsNegativeInvestment)
                //    || (InvestmentType == BuySell.Sell && dataSetAnalyzes.First().IsPositiveInvestment))
                //{
                //    IsEndOfInvestment = true;
                //    Action = Action.Released;
                //    ActionReason = ActionReason.PredictionInverse;
                //    return;
                //}
                //else if ((InvestmentType == BuySell.Buy && dataSetAnalyzes.First().IsPositiveInvestment)
                //    || (InvestmentType == BuySell.Sell && dataSetAnalyzes.First().IsNegativeInvestment))
                //{
                //    Action = Action.Continued;
                //    ActionReason = ActionReason.SamePredictions;
                //    CountDay = day;
                //    PredictedChange = dataSetAnalyzes.First().PredictedChange;
                //    return;
                //}
            }

            if (daysLeft == 0)
            {
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfPeriod;
                return;
            }

            //ActionReason releaseReason = IsTimeToRelease(day);
            //if (releaseReason != ActionReason.NoReason)
            //{
            //    Action = ActionType.Released;
            //    ActionReason = releaseReason;
            //    return;
            //}
        }

        internal object GetLiveLength(DateTime day)
        {
            return DataSet.GetDayNum(InvestmentDay) - DataSet.GetDayNum(day);
        }

        public double UpdateAccountOnRelease(DateTime day, double balance)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return balance + GetReleaseMoney(day);
            }
            else
            {
                return balance - GetReleaseMoney(day);
            }
        }

        public double UpdateAccountOnInvestment(DateTime day, double balance)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return balance - GetInvestmentMoney(DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Buy);
            }
            else
            {
                return balance + GetInvestmentMoney(DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Sell);
            }
        }

        public void UpdateRealMoneyOnInvestment(DateTime day, ref double realMoney)
        {
            realMoney -= SGSettings.SafesForStockRate * InvestedMoney;
        }

        public void UpdateRealMoneyOnRelease(DateTime day, ref double realMoney)
        {
            realMoney += SGSettings.SafesForStockRate * InvestedMoney + GetProfit(day);
        }

        public double GetReleaseMoney(DateTime day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            return GetInvestmentMoney(DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
        }

        public double GetProfit(DateTime day)
        {
            return (InvestmentType == BuySell.Buy) ? GetReleaseMoney(day) - InvestedMoney : InvestedMoney - GetReleaseMoney(day);
        }

        public double Release(DateTime day, double stockTotalProfit)
        {
            StockTotalProfit = stockTotalProfit + GetProfit(day);
            ReleaseDay = day;
            Status = InvestmentStatus.Released;

            return StockTotalProfit;
        }

        #endregion

        #region Private Methods

        private ActionReason IsTimeToRelease(DateTime day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            double releaseMoney = GetInvestmentMoney(DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
            double profitRatio = (InvestmentType == BuySell.Buy) ? (GetReleaseMoney(day) - InvestedMoney) / InvestedMoney : (InvestedMoney - GetReleaseMoney(day)) / InvestedMoney;
            if (profitRatio > SGSettings.MinProfitRatio)
            {
                return ActionReason.GoodProfit;
            }

            if (profitRatio < SGSettings.MaxLooseRatio)
            {
                return ActionReason.BadLoose;
            }

            return ActionReason.NoReason;
        }

        public double GetInvestmentMoney(double price, BuySell buyOrSell)
        {
            if (buyOrSell == BuySell.Buy)
            {
                return Ammount * (price + price * SGSettings.BuySellPenalty);
            }
            else
            {
                return Ammount * (price - price * SGSettings.BuySellPenalty);
            }
        }

        #endregion

    }
}
