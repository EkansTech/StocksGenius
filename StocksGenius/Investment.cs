using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using StocksData;

namespace StocksGenius
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

        public DateTime InvestmentDate { get; set; }

        public double InvestedPrice { get; set; }

        public double InvestedMoney { get; set; }

        public double AccountBefore { get; set; }

        public Analyze Analyze { get; set; }

        public bool IsEndOfInvestment { get; set; }

        public BuySell InvestmentType { get; set; }

        public double Profit { get; set; }

        public double ReleasePrice { get; set; }

        public double AccountAfter { get; set; }

        #endregion

        #region Constructors

        public Investment(Analyze analyze, DateTime date, double investmentPrice, double accountBefore, int ammount, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            DataSet = analyze.DataSet;
            PredictedChange = analyze.PredictedChange;
            InvestmentDate = date;
            InvestedPrice = investmentPrice;
            Ammount = ammount;
            AccountBefore = accountBefore;
            Analyze = analyze;
            IsEndOfInvestment = false;
            InvestmentType = (analyze.IsPositiveInvestment) ? BuySell.Buy : BuySell.Sell;
            InvestedMoney = ammount * investmentPrice;
        }

        #endregion

        #region Interface

        #endregion

        #region Private Methods

        #endregion

    }
}
