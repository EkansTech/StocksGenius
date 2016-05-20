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

        public double InvestmentValue { get; set; }

        public double TotalValue { get; set; }

        public Analyze Analyze { get; set; }

        public bool IsEndOfInvestment { get; set; }

        public BuySell InvestmentType { get; set; }

        public int ReleaseDay { get; set; }

        public ActionType Action { get; set; }

        public ActionReason ActionReason { get; set; }

        public double StockTotalProfit { get; set; }

        public bool OnLooseSaving { get; set; }

        public double RealMoney { get; set; }

        #endregion

        #region Constructors

        public Investment(DataSet dataSet, DataSet priceDataSet, Analyze analyze, DateTime day, double totalValue, double realMoney, double stockTotalProfit, double addPercentagePrice, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            DataSet = dataSet;
            PriceDataSet = priceDataSet;
            DataSet = analyze.DataSet;
            PredictedChange = analyze.PredictedChange;
            CountDay = dataSet.GetDayNum(day);
            InvestmentDay = dataSet.GetDayNum(day);
            InvestedPrice = PriceDataSet.GetDayData(day)[(int)dataColumn] * (1 + addPercentagePrice);
            Ammount = (int)(SimSettings.InvestmentPerStock / InvestedPrice);
            TotalValue = totalValue;
            RealMoney = realMoney;
            Analyze = analyze;
            IsEndOfInvestment = false;
            InvestmentType = (analyze.IsPositiveInvestment) ? BuySell.Buy : BuySell.Sell; // Test opposite decision
            InvestedMoney = GetInvestmentMoney(InvestedPrice, InvestmentType);
            Action = ActionType.Created;
            ActionReason = ActionReason.NoReason;
            OnLooseSaving = false;
            StockTotalProfit = stockTotalProfit;
            InvestmentValue = GetInvestmentValue(day);
        }

        public Investment(DataSet dataSet, DateTime day, double totalValue, double totalProfit, double stockTotalProfit, BuySell investmentType, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            PriceDataSet = dataSet;
            DataSet = dataSet;
            //PredictedChange = null;
            CountDay = dataSet.GetDayNum(day);
            InvestmentDay = dataSet.GetDayNum(day);
            InvestedPrice = PriceDataSet.GetDayData(day)[(int)dataColumn];
            Ammount = (int)(SimSettings.InvestmentPerStock / InvestedPrice);
            TotalValue = totalValue;
            Analyze = null;
            IsEndOfInvestment = false;
            InvestmentType = investmentType;
            InvestedMoney = GetInvestmentMoney(InvestedPrice, InvestmentType);
            Action = ActionType.Created;
            ActionReason = ActionReason.NoReason;
            OnLooseSaving = false;
            StockTotalProfit = stockTotalProfit;
            InvestmentValue = GetInvestmentValue(day);
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
            TotalValue = investment.TotalValue;
            RealMoney = investment.RealMoney;
            Analyze = investment.Analyze;
            IsEndOfInvestment = investment.IsEndOfInvestment;
            InvestmentType = investment.InvestmentType;
            InvestedMoney = investment.InvestedMoney;
            Action = investment.Action;
            ActionReason = investment.ActionReason;
            OnLooseSaving = investment.OnLooseSaving;
            Profit = investment.Profit;
            InvestmentValue = investment.InvestmentValue;
            TotalValue = investment.TotalValue;
            StockTotalProfit = investment.StockTotalProfit;
            ReleaseDay = investment.ReleaseDay;
        }

        #endregion

        #region Interface

        internal static void Reset()
        {
            m_IDs = 0;
        }

        internal double ProfitPercentage(DateTime day)
        {
            return Profit / InvestedMoney * 100;
        }

        internal double GetDayPrice(DateTime day)
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

        public void UpdateInvestment(DateTime day, double totalValue, double stockTotalProfit)
        {
            Action = ActionType.NoAction;
            TotalValue = totalValue;
            StockTotalProfit = stockTotalProfit;
            Profit = GetProfit(day);
            InvestmentValue = GetInvestmentValue(day);

            if (DataSet.GetDayNum(day) == 0)
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

            if (InvestmentDay - DataSet.GetDayNum(day) >= 1)
            {
                IsEndOfInvestment = true;
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfPeriod;
            }

        }

        public void UpdateInvestment(DailyAnalyzes dailyAnalyzes, DateTime day, double totalValue, double realMoney, double stockTotalProfit)
        {
            Action = ActionType.NoAction;
            TotalValue = totalValue;
            StockTotalProfit = stockTotalProfit;
            Profit = GetProfit(day);
            InvestmentValue = GetInvestmentValue(day);
            ReleaseDay = DataSet.GetDayNum(day);
            RealMoney = realMoney;
            ActionReason releaseReason;

            //if (InvestmentDay - day >= AnalyzerSimulator.MaxInvestmentsLive)
            //{
            //    Action = Action.Released;
            //    ActionReason = ActionReason.MaxInvestmentLive;
            //    IsEndOfInvestment = true;
            //    return;
            //}

            if (DataSet.GetDayNum(day) == 0)
            {
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfTrade;
                IsEndOfInvestment = true;
                return;
            }

            int daysLeft = PredictedChange.Range - (CountDay - DataSet.GetDayNum(day));

            if (dailyAnalyzes.ContainsKey(DataSet))
            {
                var dataSetAnalyzes = dailyAnalyzes[DataSet].Values.OrderBy(x => x.PredictedChange.Range);
                if ((InvestmentType == BuySell.Buy && dataSetAnalyzes.First().IsNegativeInvestment)
                    || (InvestmentType == BuySell.Sell && dataSetAnalyzes.First().IsPositiveInvestment))
                {
                    IsEndOfInvestment = true;
                    Action = ActionType.Released;
                    ActionReason = ActionReason.PredictionInverse;
                    return;
                }
                else if ((InvestmentType == BuySell.Buy && dataSetAnalyzes.First().IsPositiveInvestment)
                    || (InvestmentType == BuySell.Sell && dataSetAnalyzes.First().IsNegativeInvestment))
                {
                    releaseReason = IsTimeToRelease(DataSet.GetDayNum(day));
                    if (releaseReason != ActionReason.NoReason)
                    {
                        IsEndOfInvestment = true;
                        Action = ActionType.Released;
                        ActionReason = releaseReason;
                        return;
                    }
                    
                    Action = ActionType.Continued;
                    ActionReason = ActionReason.SamePredictions;
                    CountDay = DataSet.GetDayNum(day);
                    PredictedChange = dataSetAnalyzes.First().PredictedChange;
                    return;
                }
            }

            releaseReason = IsTimeToRelease(DataSet.GetDayNum(day));
            if (releaseReason != ActionReason.NoReason)
            {
                IsEndOfInvestment = true;
                Action = ActionType.Released;
                ActionReason = releaseReason;
                return;
            }

            if (daysLeft <= 0)
            {
                IsEndOfInvestment = true;
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfPeriod;
                return;
            }

            //if (profit > 0)
            //{
            //    isendofinvestment = true;
            //    action = actiontype.released;
            //    actionreason = actionreason.goodprofit;
            //    return;
            //}
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
            Profit = GetProfit(day);
            InvestmentValue = GetInvestmentValue(day);
            if (InvestmentType == BuySell.Buy)
            {
                return balance - GetInvestmentMoney(PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Buy);
            }
            else
            {
                return balance + GetInvestmentMoney(PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Sell);
            }
        }

        public void UpdateRealMoneyOnInvestment(DateTime day, ref double realMoney)
        {
            realMoney -= SimSettings.SafesForStockRate * InvestedMoney;
        }

        public void UpdateRealMoneyOnRelease(DateTime day, ref double realMoney)
        {
            realMoney += SimSettings.SafesForStockRate * InvestedMoney + Profit;
        }

        public double GetReleaseMoney(DateTime day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            return GetInvestmentMoney(PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
        }

        public double GetProfit(DateTime day)
        {
            return (InvestmentType == BuySell.Buy) ? GetReleaseMoney(day) - InvestedMoney : InvestedMoney - GetReleaseMoney(day);
        }

        public double GetInvestmentValue(DateTime day)
        {
            return SimSettings.SafesForStockRate * InvestedMoney + GetProfit(day);
        }

        public double GetInvestmentValue(DateTime day, int offset)
        {
            DateTime offsetDay = DataSet.GetDate(DataSet.GetDayNum(day) + 1);
            return SimSettings.SafesForStockRate * InvestedMoney + GetProfit(offsetDay);
        }

        public double Release(DateTime day, ref double totalProfit, double stockTotalProfit)
        {
            Profit = GetProfit(day);
            InvestmentValue = GetInvestmentValue(day);
            totalProfit += Profit;
            TotalValue = totalProfit;
            StockTotalProfit = stockTotalProfit + Profit;
            ReleaseDay = DataSet.GetDayNum(day);

            return StockTotalProfit;
        }

        public double Release(DateTime day, double totalValue, double stockTotalProfit)
        {
            Profit = GetProfit(day);
            InvestmentValue = GetInvestmentValue(day);
            TotalValue = totalValue;
            StockTotalProfit = stockTotalProfit + Profit;
            ReleaseDay = DataSet.GetDayNum(day);

            return StockTotalProfit;
        }

        #endregion

        #region Private Methods

        private ActionReason IsTimeToRelease(int day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            double releaseMoney = GetInvestmentMoney(PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
            double profitRatio = Profit / InvestedMoney;
            if (profitRatio > PredictionsSimulator.MinProfitRatio)
            {
                //return ActionReason.GoodProfit;
            }

            //if (OnLooseSaving && profitRatio > AnalyzerSimulator.MaxLooseRatio / 2)
            //{
            //    OnLooseSaving = false;
            //    return ActionReason.BadLoose;
            //}

            if (profitRatio < PredictionsSimulator.MaxLooseRatio)// && (CountDay - day) < PredictedChange.Range / 2)
            {
                //OnLooseSaving = true;
                //return ActionReason.BadLoose;
            }

            return ActionReason.NoReason;
        }

        private double GetInvestmentMoney(double price, BuySell buyOrSell)
        {
            if (buyOrSell == BuySell.Buy)
            {
                return Ammount * (price + price * SimSettings.BuySellPenalty);
            }
            else
            {
                return Ammount * (price - price * SimSettings.BuySellPenalty);
            }
        }

        #endregion

    }
}
