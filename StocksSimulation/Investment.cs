using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using StocksData;

namespace StocksSimulation
{
    public enum BuySell
    {
        Buy,
        Sell
    }

    public enum ReleaseReason
    {
        NoReason,
        EndOfPeriod,
        PredictionInverse,
        GoodProfit,
        BadLoose,
    }

    public class Investment
    {
        #region Properties

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

        public ReleaseReason ReleaseReason { get; set; }

        public double ReleaseTotalProfit { get; set; }

        public bool OnLooseSaving { get; set; }

        #endregion

        #region Constructors

        public Investment(DataSet priceDataSet, Analyze analyze, int day, double accountBefore, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
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
            ReleaseReason = ReleaseReason.NoReason;
            OnLooseSaving = false;
        }

        #endregion

        #region Interface

        public void UpdateInvestment(DailyAnalyzes dailyAnalyzes, int day)
        {
            if (day == 0)
            {
                IsEndOfInvestment = true;
            }

            int daysLeft = PredictedChange.Range - (CountDay - day);

            if (dailyAnalyzes.ContainsKey(DataSet))
            {
                var dataSetAnalyzes = dailyAnalyzes[DataSet].Values.OrderBy(x => x.PredictedChange.Range);
                if ((InvestmentType == BuySell.Buy && dataSetAnalyzes.First().IsNegativeInvestment)
                    || (InvestmentType == BuySell.Sell && dataSetAnalyzes.First().IsPositiveInvestment))
                {
                    IsEndOfInvestment = true;
                    ReleaseReason = ReleaseReason.PredictionInverse;
                    return;
                }
                else if ((InvestmentType == BuySell.Buy && dataSetAnalyzes.First().IsPositiveInvestment)
                    || (InvestmentType == BuySell.Sell && dataSetAnalyzes.First().IsNegativeInvestment))
                {
                    CountDay = day;
                    PredictedChange = dataSetAnalyzes.First().PredictedChange;
                }
            }
            else if (daysLeft == 0)
            {
                IsEndOfInvestment = true;
                ReleaseReason = ReleaseReason.EndOfPeriod;
                return;
            }

            ReleaseReason releaseReason = IsTimeToRelease(day);
            if (releaseReason != ReleaseReason.NoReason)
            {
                IsEndOfInvestment = true;
                ReleaseReason = releaseReason;
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
            return (InvestmentType == BuySell.Buy) ? GetReleasePrice(day) - InvestedMoney : InvestedMoney - GetReleasePrice(day); ;
        }

        public double Release(int day, double totalProfit)
        {
            Profit = GetProfit(day);
            totalProfit += Profit;
            ReleaseTotalProfit = totalProfit;
            ReleaseDay = day;

            return totalProfit;
        }

        #endregion

        #region Private Methods

        private ReleaseReason IsTimeToRelease(int day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            double releaseMoney = GetInvestmentMoney(Ammount, PriceDataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
            double profitRatio = (InvestmentType == BuySell.Buy) ? (GetReleasePrice(day) - InvestedMoney) / InvestedMoney : (InvestedMoney - GetReleasePrice(day)) / InvestedMoney;
            if (profitRatio > AnalyzerSimulator.MinProfitRatio)
            {
                return ReleaseReason.GoodProfit;
            }

            //if (OnLooseSaving && profitRatio > AnalyzerSimulator.MaxLooseRatio / 2)
            //{
            //    OnLooseSaving = false;
            //    return ReleaseReason.BadLoose;
            //}

            if (profitRatio < AnalyzerSimulator.MaxLooseRatio && CountDay - day < PredictedChange.Range / 1.5)
            {
                OnLooseSaving = true;
                return ReleaseReason.BadLoose;
            }

            return ReleaseReason.NoReason;
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
