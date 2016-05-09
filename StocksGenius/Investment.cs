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

        public DataSet PriceDataSet
        {
            get
            {
                return Investor.StocksData.PriceDataSets[DataSetName];
            }
        }

        public DataSet DataSet
        {
            get
            {
                return Investor.StocksData.DataSets[DataSetName];
            }
        }

        public string DataSetName { get; set; }

        public int Ammount { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public Analyze Analyze { get; set; }

        public DateTime InvestmentDay { get; set; }

        public DateTime ReleaseDay { get; set; }

        public double InvestedPrice { get; set; }

        public double ReleasePrice { get; set; }

        public double InvestedMoney { get; set; }

        public double Profit { get; set; }

        public double AccountBefore { get; set; }

        public BuySell InvestmentType { get; set; }

        public ActionType Action { get; set; }

        public ActionReason ActionReason { get; set; }

        public double TotalProfit { get; set; }

        public double StockTotalProfit { get; set; }

        public InvestmentStatus Status { get; set; }

        #endregion

        #region Constructors

        public Investment(int id)
        {
            ID = id;
        }

        public Investment(Analyze analyze, DateTime day, double accountBefore, double totalProfit, double stockTotalProfit, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            DataSetName = analyze.DataSetName;
            PredictedChange = analyze.PredictedChange;
            InvestmentDay = day;
            InvestedPrice = PriceDataSet.GetDayData(day)[(int)dataColumn];
            Ammount = (int)(SGSettings.InvestmentPerStock / InvestedPrice);
            AccountBefore = accountBefore;
            Analyze = analyze;
            InvestmentType = (analyze.IsPositiveInvestment) ? BuySell.Buy : BuySell.Sell; // Test opposite decision
            InvestedMoney = GetInvestmentMoney(InvestedPrice, InvestmentType);
            Action = ActionType.Created;
            ActionReason = ActionReason.NoReason;
            TotalProfit = totalProfit;
            StockTotalProfit = stockTotalProfit;
            Status = InvestmentStatus.Active;
        }

        public Investment(string dataSetName, DateTime day, double accountBefore, double totalProfit, double stockTotalProfit, BuySell investmentType, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            DataSetName = dataSetName;
            //PredictedChange = null;
            InvestmentDay = day;
            InvestedPrice = PriceDataSet.GetDayData(day)[(int)dataColumn];
            Ammount = (int)(SGSettings.InvestmentPerStock / InvestedPrice);
            AccountBefore = accountBefore;
            Analyze = null;
            InvestmentType = investmentType;
            InvestedMoney = GetInvestmentMoney(InvestedPrice, InvestmentType);
            Action = ActionType.Created;
            ActionReason = ActionReason.NoReason;
            TotalProfit = totalProfit;
            StockTotalProfit = stockTotalProfit;
            Status = InvestmentStatus.Active;
        }

        private Investment(Investment investment)
        {
            ID = investment.ID;
            DataSetName = investment.DataSetName;
            PredictedChange = investment.PredictedChange;
            InvestmentDay = investment.InvestmentDay;
            InvestedPrice = investment.InvestedPrice;
            Ammount = investment.Ammount;
            AccountBefore = investment.AccountBefore;
            Analyze = investment.Analyze;
            InvestmentType = investment.InvestmentType;
            InvestedMoney = investment.InvestedMoney;
            Action = investment.Action;
            ActionReason = investment.ActionReason;
            Profit = investment.Profit;
            TotalProfit = investment.TotalProfit;
            StockTotalProfit = investment.StockTotalProfit;
            ReleaseDay = investment.ReleaseDay;
            Status = InvestmentStatus.Active;
        }

        #endregion

        #region Interface

        internal static void Reset(int id, int releaseId)
        {
            m_IDs = id;
            m_ReleaseIDs = releaseId;
        }

        internal double CurrentProfitPercentage()
        {
            return Profit / InvestedMoney * 100;
        }

        internal double GetDayPrice(int day)
        {
            return PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open];
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
        
        public void UpdateInvestment(DailyAnalyzes dailyAnalyzes, DateTime day, double totalProfit, double stockTotalProfit)
        {
            Action = ActionType.NoAction;
            TotalProfit = totalProfit;
            StockTotalProfit = stockTotalProfit;
            Profit = GetProfit();

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

            if (dailyAnalyzes.ContainsKey(DataSetName))
            {
                var dataSetAnalyzes = dailyAnalyzes[DataSetName].Values.OrderBy(x => x.PredictedChange.Range);
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

            ActionReason releaseReason = IsTimeToRelease(day);
            if (releaseReason != ActionReason.NoReason)
            {
                Action = ActionType.Released;
                ActionReason = releaseReason;
                return;
            }
        }

        public double UpdateAccountOnRelease(double balance)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return balance + (GetReleaseMoney() - InvestedMoney);
            }
            else
            {
                return balance - (GetReleaseMoney() - InvestedMoney);
            }
        }

        public double UpdateAccountOnInvestment(double balance)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return balance - GetInvestmentMoney(PriceDataSet.GetDayData()[(int)DataSet.DataColumns.Open], BuySell.Buy);
            }
            else
            {
                return balance + GetInvestmentMoney(PriceDataSet.GetDayData()[(int)DataSet.DataColumns.Open], BuySell.Sell);
            }
        }

        public void UpdateRealMoneyOnInvestment(ref double realMoney)
        {
            realMoney -= SGSettings.SafesForStockRate * GetInvestmentMoney(PriceDataSet.GetDayData()[(int)DataSet.DataColumns.Open], InvestmentType);
        }

        public void UpdateRealMoneyOnRelease(ref double realMoney)
        {
            realMoney += SGSettings.SafesForStockRate * GetReleaseMoney() + Profit;
        }

        public double GetReleaseMoney()
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            return GetInvestmentMoney(PriceDataSet.GetDayData(0)[(int)DataSet.DataColumns.Open], releaseAction);
        }

        public double GetReleaseMoney(int day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            return GetInvestmentMoney(PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
        }

        public double GetProfit()
        {
            return (InvestmentType == BuySell.Buy) ? GetReleaseMoney() - InvestedMoney : InvestedMoney - GetReleaseMoney();
        }

        public double GetCurrentProfit()
        {
            return (InvestmentType == BuySell.Buy) ? GetReleaseMoney() - InvestedMoney : InvestedMoney - GetReleaseMoney();
        }

        public double Release(ref double totalProfit, double stockTotalProfit)
        {
            Profit = GetProfit();
            totalProfit += Profit;
            TotalProfit = totalProfit;
            StockTotalProfit = stockTotalProfit + Profit;
            ReleaseDay = DataSet.GetDate(0);
            Status = InvestmentStatus.Released;
            ReleaseID = m_ReleaseIDs++;

            return StockTotalProfit;
        }

        public static double GetInvestmentPrice(DataSet priceDataSet, bool isPositiveInvestment)
        {
            return priceDataSet.GetData(0, DataSet.DataColumns.Open) * (1 + (isPositiveInvestment ? SGSettings.BuySellPenalty : -SGSettings.BuySellPenalty));
        } 

        public double GetInvestmentValue()
        {
            if (InvestmentType == BuySell.Buy)
            {
                return SGSettings.SafesForStockRate * InvestedMoney + (GetReleaseMoney() - InvestedMoney);
            }
            else
            {
                return SGSettings.SafesForStockRate * InvestedMoney + (InvestedMoney - GetReleaseMoney());
            }
        }

        internal int GetLiveLength()
        {
            return DataSet.GetDayNum(InvestmentDay);
        }

        #endregion

        #region Private Methods

        private ActionReason IsTimeToRelease(DateTime day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            double releaseMoney = GetInvestmentMoney(PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
            double profitRatio = (InvestmentType == BuySell.Buy) ? (GetReleaseMoney() - InvestedMoney) / InvestedMoney : (InvestedMoney - GetReleaseMoney()) / InvestedMoney;
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

        private double GetInvestmentMoney(double price, BuySell buyOrSell)
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
