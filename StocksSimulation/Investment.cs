using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using StocksData;

namespace StocksSimulation
{
    internal class Investment
    {
        #region Members

        private static int m_IDs = 0;

        #endregion

        #region Properties

        public int ID { get; private set; }

        public DataSet PriceDataSet { get; set; }

        public DataSet DataSet { get; set; }

        public int Ammount { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public int CountDay { get; set; }

        public int InvestmentDay { get; set; }

        public double InvestedPrice { get; set; }

        public double InvestedMoney { get; set; }

        public double Profit { get; set; }

        public double AccountBefore { get; set; }

        public Analyze Analyze { get; set; }

        public bool IsEndOfInvestment { get; set; }

        public BuySell InvestmentType { get; set; }

        public int ReleaseDay { get; set; }

        public ActionType Action { get; set; }

        public ActionReason ActionReason { get; set; }

        public double ReleaseTotalProfit { get; set; }

        public double ReleaseStockTotalProfit { get; set; }

        public bool OnLooseSaving { get; set; }

        #endregion

        #region Constructors

        public Investment(DataSet dataSet, DataSet priceDataSet, Analyze analyze, int day, double accountBefore, double totalProfit, double stockTotalProfit, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            DataSet = dataSet;
            PriceDataSet = priceDataSet;
            DataSet = analyze.DataSet;
            PredictedChange = analyze.PredictedChange;
            CountDay = day;
            InvestmentDay = day;
            InvestedPrice = PriceDataSet.GetDayData(day)[(int)dataColumn];
            Ammount = (int)(SimSettings.InvestmentMoney / InvestedPrice);
            AccountBefore = accountBefore;
            Analyze = analyze;
            IsEndOfInvestment = false;
            InvestmentType = (analyze.IsPositiveInvestment) ? BuySell.Buy : BuySell.Sell; // Test opposite decision
            InvestedMoney = GetInvestmentMoney(Ammount, InvestedPrice, InvestmentType);
            Action = ActionType.Created;
            ActionReason = ActionReason.NoReason;
            OnLooseSaving = false;
            ReleaseTotalProfit = totalProfit;
            ReleaseStockTotalProfit = stockTotalProfit;
        }

        public Investment(DataSet dataSet, int day, double accountBefore, double totalProfit, double stockTotalProfit, BuySell investmentType, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            PriceDataSet = dataSet;
            DataSet = dataSet;
            //PredictedChange = null;
            CountDay = day;
            InvestmentDay = day;
            InvestedPrice = PriceDataSet.GetDayData(day)[(int)dataColumn];
            Ammount = (int)(SimSettings.InvestmentMoney / InvestedPrice);
            AccountBefore = accountBefore;
            Analyze = null;
            IsEndOfInvestment = false;
            InvestmentType = investmentType;
            InvestedMoney = GetInvestmentMoney(Ammount, InvestedPrice, InvestmentType);
            Action = ActionType.Created;
            ActionReason = ActionReason.NoReason;
            OnLooseSaving = false;
            ReleaseTotalProfit = totalProfit;
            ReleaseStockTotalProfit = stockTotalProfit;
        }

        private Investment(Investment investment)
        {
            ID = investment.ID;
            PriceDataSet = investment.PriceDataSet;
            DataSet = investment.DataSet;
            PredictedChange = investment.PredictedChange;
            CountDay = investment.CountDay;
            InvestmentDay = investment.InvestmentDay;
            InvestedPrice = investment.InvestedPrice;
            Ammount = investment.Ammount;
            AccountBefore = investment.AccountBefore;
            Analyze = investment.Analyze;
            IsEndOfInvestment = investment.IsEndOfInvestment;
            InvestmentType = investment.InvestmentType;
            InvestedMoney = investment.InvestedMoney;
            Action = investment.Action;
            ActionReason = investment.ActionReason;
            OnLooseSaving = investment.OnLooseSaving;
            Profit = investment.Profit;
            ReleaseTotalProfit = investment.ReleaseTotalProfit;
            ReleaseStockTotalProfit = investment.ReleaseStockTotalProfit;
            ReleaseDay = investment.ReleaseDay;
        }

        #endregion

        #region Interface

        internal static void Reset()
        {
            m_IDs = 0;
        }

        internal double CurrentProfitPercentage(int day)
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
            if (dataSet.GetContinuousChange(day, DataSet.DataColumns.Open, 1) > StockSimulation.MinimumChange)
            {
                return BuySell.Buy;
            }
            else if (dataSet.GetContinuousChange(day, DataSet.DataColumns.Open, 1) < -StockSimulation.MinimumChange)
            {
                return BuySell.Sell;
            }

            return BuySell.None;
        }

        public void UpdateInvestment(int day, double totalProfit, double stockTotalProfit)
        {
            Action = ActionType.NoAction;
            ReleaseTotalProfit = totalProfit;
            ReleaseStockTotalProfit = stockTotalProfit;
            Profit = GetProfit(day);

            if (day == 0)
            {
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfTrade;
                IsEndOfInvestment = true;
                return;
            }

            if (InvestmentType == BuySell.Sell)
            {
                if (DataSet.GetLastChange(day, DataSet.DataColumns.Open) > 0)
                {
                    Action = ActionType.Continued;
                    ActionReason = ActionReason.SamePredictions;
                    return;
                }
            }
            else if (InvestmentType == BuySell.Buy)
            {
                if (DataSet.GetLastChange(day, DataSet.DataColumns.Open) < 0)
                {
                    Action = ActionType.Continued;
                    ActionReason = ActionReason.SamePredictions;
                    return;
                }
            }

            if (InvestmentDay - day >= 1)
            {
                IsEndOfInvestment = true;
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfPeriod;
            }

        }

        public void UpdateInvestment(DailyAnalyzes dailyAnalyzes, int day, double totalProfit, double stockTotalProfit)
        {
            Action = ActionType.NoAction;
            ReleaseTotalProfit = totalProfit;
            ReleaseStockTotalProfit = stockTotalProfit;
            Profit = GetProfit(day);

            //if (InvestmentDay - day >= AnalyzerSimulator.MaxInvestmentsLive)
            //{
            //    Action = Action.Released;
            //    ActionReason = ActionReason.MaxInvestmentLive;
            //    IsEndOfInvestment = true;
            //    return;
            //}

            if (day == 0)
            {
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfTrade;
                IsEndOfInvestment = true;
            }

            int daysLeft = PredictedChange.Range - (CountDay - day);

            if (dailyAnalyzes.ContainsKey(DataSet))
            {
                var dataSetAnalyzes = dailyAnalyzes[DataSet].Values.OrderBy(x => x.PredictedChange.Range);
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
                IsEndOfInvestment = true;
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfPeriod;
                return;
            }

            ActionReason releaseReason = IsTimeToRelease(day);
            if (releaseReason != ActionReason.NoReason)
            {
                IsEndOfInvestment = true;
                Action = ActionType.Released;
                ActionReason = releaseReason;
                return;
            }
        }

        public double UpdateAccountOnRelease(int day, double balance)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return balance + GetReleasePrice(day);
            }
            else
            {
                return balance - GetReleasePrice(day);
            }
        }

        public double UpdateAccountOnInvestment(int day, double balance)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return balance - GetInvestmentMoney(Ammount, PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Buy);
            }
            else
            {
                return balance + GetInvestmentMoney(Ammount, PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Sell);
            }
        }

        public double GetReleasePrice(int day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            return GetInvestmentMoney(Ammount, PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
        }

        public double GetProfit(int day)
        {
            return (InvestmentType == BuySell.Buy) ? GetReleasePrice(day) - InvestedMoney : InvestedMoney - GetReleasePrice(day);
        }

        public double Release(int day, ref double totalProfit, double stockTotalProfit)
        {
            Profit = GetProfit(day);
            totalProfit += Profit;
            ReleaseTotalProfit = totalProfit;
            ReleaseStockTotalProfit = stockTotalProfit + Profit;
            ReleaseDay = day;

            return ReleaseStockTotalProfit;
        }

        #endregion

        #region Private Methods

        private ActionReason IsTimeToRelease(int day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            double releaseMoney = GetInvestmentMoney(Ammount, PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
            double profitRatio = (InvestmentType == BuySell.Buy) ? (GetReleasePrice(day) - InvestedMoney) / InvestedMoney : (InvestedMoney - GetReleasePrice(day)) / InvestedMoney;
            if (profitRatio > PredictionsSimulator.MinProfitRatio)
            {
                return ActionReason.GoodProfit;
            }

            //if (OnLooseSaving && profitRatio > AnalyzerSimulator.MaxLooseRatio / 2)
            //{
            //    OnLooseSaving = false;
            //    return ActionReason.BadLoose;
            //}

            if (profitRatio < PredictionsSimulator.MaxLooseRatio && (CountDay - day) < PredictedChange.Range / 2)
            {
                OnLooseSaving = true;
                return ActionReason.BadLoose;
            }

            return ActionReason.NoReason;
        }

        private double GetInvestmentMoney(int ammount, double price, BuySell buyOrSell)
        {
            if (buyOrSell == BuySell.Buy)
            {
                return ammount * (price + price * SimSettings.BuySellPenalty);
            }
            else
            {
                return ammount * (price - price * SimSettings.BuySellPenalty);
            }
        }

        #endregion

    }
}
