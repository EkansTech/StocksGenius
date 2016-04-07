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
    public class Investment
    {
        #region Properties

        public DataSet DataSet { get; set; }

        public int Ammount { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public int InvestedDay { get; set; }

        public float InvestedPrice { get; set; }

        public float InvestedMoney { get; set; }

        public float AccountBefore { get; set; }

        public Analyze Analyze { get; set; }

        public bool IsEndOfInvestment { get; set; }

        public BuySell InvestmentType { get; set; }

        #endregion

        #region Constructors

        public Investment(Analyze analyze, int day, float accountBefore, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            DataSet = analyze.DataSet;
            PredictedChange = analyze.PredictedChange;
            InvestedDay = day;
            InvestedPrice = DataSet.GetDayData(day)[(int)dataColumn];
            Ammount = (int)(SimSettings.InvestmentMoney / InvestedPrice);
            AccountBefore = accountBefore;
            Analyze = analyze;
            IsEndOfInvestment = false;
            InvestmentType = (analyze.IsPositiveInvestment) ? BuySell.Buy : BuySell.Sell;
            InvestedMoney = GetInvestmentMoney(Ammount, InvestedPrice, InvestmentType);
        }

        #endregion

        #region Interface

        public void UpdateInvestment(DailyAnalyzes dailyAnalyzes, int day)
        {
            if (day == 0)
            {
                IsEndOfInvestment = true;
            }

            int daysLeft = PredictedChange.Range - (InvestedDay - day);
            
            if (daysLeft == 0)
            {
                if (dailyAnalyzes.ContainsKey(DataSet) && 
                    ((Ammount > 0 && dailyAnalyzes[DataSet].ContainsPositiveInvestmens)
                    || (Ammount < 0 && dailyAnalyzes[DataSet].ContainsNegativeInvestmens)))
                {
                    InvestedDay = day;
                }
                else
                {
                    IsEndOfInvestment = true;
                }
            }

            if (IsTimeToRelease(day))
            {
                IsEndOfInvestment = true;
            }
        }

        public float UpdateAccountOnRelease(int day, float balance)
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

        public float UpdateAccountOnInvestment(int day, float balance)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return balance - GetInvestmentMoney(Ammount, DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Buy);
            }
            else
            {
                return balance + GetInvestmentMoney(Ammount, DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Sell);
            }
        }

        public float GetReleasePrice(int day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            return GetInvestmentMoney(Ammount, DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
        }

        public float GetProfit(int day)
        {
            return (InvestmentType == BuySell.Buy) ? GetReleasePrice(day) - InvestedMoney : InvestedMoney - GetReleasePrice(day); ;
        }

        #endregion

        #region Private Methods

        private bool IsTimeToRelease(int day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            float releaseMoney = GetInvestmentMoney(Ammount, DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
            float profitRatio = (InvestmentType == BuySell.Buy) ? (GetReleasePrice(day) - InvestedMoney) / InvestedMoney : (InvestedMoney - GetReleasePrice(day)) / InvestedMoney;
            if (profitRatio > AnalyzerSimulator.MinProfitRatio || profitRatio < AnalyzerSimulator.MaxLooseRatio)
            {
                return true;
            }

            return false;
        }

        private float GetInvestmentMoney(int ammount, float price, BuySell buyOrSell)
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
