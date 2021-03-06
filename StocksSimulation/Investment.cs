﻿using System;
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

        public DataSet DataSet { get; set; }

        public double Ammount { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public int CountDay { get; set; }

        public int InvestmentDay { get; set; }

        public double InvestedPrice { get; set; }

        public double InvestedMoney { get; set; }

        public double Profit { get; set; }

        public double InvestmentValue { get; set; }

        public double TotalValue { get; set; }
        
        public double TotalProfit { get; set; }

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

        public Investment(DataSet dataSet, Analyze analyze, DateTime day, double totalValue, double totalProfit, double realMoney, double stockTotalProfit, double addPercentagePrice, ActionReason openReason = ActionReason.NoReason, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            DataSet = dataSet;
            DataSet = analyze.DataSet;
            PredictedChange = analyze.PredictedChange;
            CountDay = dataSet.GetDayNum(day);
            InvestmentDay = dataSet.GetDayNum(day);
            InvestedPrice = dataSet.GetDayData(day)[(int)dataColumn] * (1 + addPercentagePrice);
            Ammount = (SimSettings.InvestmentPerStock / InvestedPrice);
            TotalValue = totalValue;
            TotalProfit = totalProfit;
            RealMoney = realMoney;
            Analyze = analyze;
            IsEndOfInvestment = false;
            InvestmentType = (analyze.IsPositiveInvestment) ? PredictionsSimulator.InvertInvestments ? BuySell.Sell : BuySell.Buy : PredictionsSimulator.InvertInvestments ? BuySell.Buy : BuySell.Sell; // Test opposite decision
            InvestedMoney = GetInvestmentMoney(InvestedPrice, InvestmentType);
            Action = ActionType.Created;
            ActionReason = openReason;
            OnLooseSaving = false;
            StockTotalProfit = stockTotalProfit;
            InvestmentValue = GetInvestmentValue(day);
        }

        public Investment(DataSet dataSet, DateTime day, double totalValue, double totalProfit, double stockTotalProfit, BuySell investmentType, DataSet.DataColumns dataColumn = DataSet.DataColumns.Open)
        {
            ID = m_IDs++;
            DataSet = dataSet;
            //PredictedChange = null;
            CountDay = dataSet.GetDayNum(day);
            InvestmentDay = dataSet.GetDayNum(day);
            InvestedPrice = dataSet.GetDayData(day)[(int)dataColumn];
            Ammount = (SimSettings.InvestmentPerStock / InvestedPrice);
            TotalValue = totalValue;
            TotalProfit = totalProfit;
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
            TotalProfit = investment.TotalProfit;
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

        public void UpdateInvestment(DateTime day, double totalValue, double totalProfit, double realMoney, double stockTotalProfit)
        {
            Action = ActionType.NoAction;
            TotalValue = totalValue;
            TotalProfit = totalProfit;
            StockTotalProfit = stockTotalProfit;
            Profit = GetProfit(day);
            InvestmentValue = GetInvestmentValue(day);
            ReleaseDay = DataSet.GetDayNum(day);
            RealMoney = realMoney;

            if (DataSet.GetDate(0) == day)
            {
                IsEndOfInvestment = true;
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfTrade;
                return;
            }

            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            double releaseMoney = GetInvestmentMoney(DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
            double profitRatio = Profit / InvestedMoney;
            //if (profitRatio >= (0.5 - (InvestmentDay - ReleaseDay) / (double)StockSimulation.MaxDaysUntilProfit) * Analyze.MinProfitRatio)
            if (profitRatio >= (1 - 2 * (InvestmentDay - ReleaseDay - 1) / (double)StockSimulation.MaxDaysUntilProfit) * Analyze.MinProfitRatio)
            //if (profitRatio >= Analyze.MinProfitRatio - (double)(InvestmentDay - ReleaseDay) / 2.0 && (InvestedMoney - ReleaseDay) > StockSimulation.MaxDaysUntilProfit)
            //if (profitRatio >= Analyze.MinProfitRatio)
            {
                IsEndOfInvestment = true;
                Action = ActionType.Released;
                ActionReason = ActionReason.GoodProfit;
                return;
            }
            
            //double change = DataSet.GetData(dayNum, DataSet.DataColumns.Open) - DataSet.GetData(dayNum + 1, DataSet.DataColumns.Open);
            //if (change != 0) //(InvestmentType == BuySell.Buy && change > 0) || (InvestmentType == BuySell.Sell && change < 0))
            //{
            //    IsEndOfInvestment = true;
            //    Action = ActionType.Released;
            //    ActionReason = ActionReason.EndOfPeriod;
            //    return;
            //}

            if (InvestmentDay - ReleaseDay >= StockSimulation.MaxDaysUntilProfit)
            {
                IsEndOfInvestment = true;
                Action = ActionType.Released;
                ActionReason = ActionReason.EndOfPeriod;
                return;
            }

            //if (profitRatio <= StockSimulation.MaxLooseRatio)
            //{
            //    IsEndOfInvestment = true;
            //    Action = ActionType.Released;
            //    ActionReason = ActionReason.BadLoose;
            //    return;
            //}
        }

        public void UpdateInvestment(DailyAnalyzes dailyAnalyzes, DateTime day, double totalValue, double totalProfit, double realMoney, double stockTotalProfit)
        {
            Action = ActionType.NoAction;
            TotalValue = totalValue;
            TotalProfit = totalProfit;
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

            if (daysLeft <= 0)
            {
                if (dailyAnalyzes.ContainsKey(DataSet))
                {
                    var dataSetAnalyzes = dailyAnalyzes[DataSet];
                    if ((PredictionsSimulator.InvertInvestments && (InvestmentType == BuySell.Sell && dataSetAnalyzes.ContainsPositiveInvestmens)
                        || (InvestmentType == BuySell.Buy && dataSetAnalyzes.ContainsNegativeInvestmens))
                        || (!PredictionsSimulator.InvertInvestments && (InvestmentType == BuySell.Buy && dataSetAnalyzes.ContainsPositiveInvestmens)
                        || (InvestmentType == BuySell.Sell && dataSetAnalyzes.ContainsNegativeInvestmens)))
                    {
                        //releaseReason = IsTimeToRelease(DataSet.GetDayNum(day));
                        //if (releaseReason != ActionReason.NoReason)
                        //{
                        //    IsEndOfInvestment = true;
                        //    Action = ActionType.Released;
                        //    ActionReason = releaseReason;
                        //    return;
                        //}

                        Action = ActionType.Continued;
                        ActionReason = ActionReason.SamePredictions;
                        CountDay = DataSet.GetDayNum(day);
                        PredictedChange = dataSetAnalyzes.Keys.First();
                        Analyze = dataSetAnalyzes[PredictedChange];
                        return;
                    }
                    IsEndOfInvestment = true;
                    Action = ActionType.Released;
                    ActionReason = ActionReason.EndOfPeriod;
                }

                return;
            }

            Action = ActionType.NoAction;

            //if (dailyAnalyzes.ContainsKey(DataSet))
            //{
            //    var dataSetAnalyzes = dailyAnalyzes[DataSet];
            //    if ((InvestmentType == BuySell.Buy && dataSetAnalyzes.ContainsNegativeInvestmens)
            //        || (InvestmentType == BuySell.Sell && dataSetAnalyzes.ContainsPositiveInvestmens))
            //    {
            //        IsEndOfInvestment = true;
            //        Action = ActionType.Released;
            //        ActionReason = ActionReason.PredictionInverse;
            //        return;
            //    }
            //    else if ((InvestmentType == BuySell.Buy && dataSetAnalyzes.ContainsPositiveInvestmens)
            //        || (InvestmentType == BuySell.Sell && dataSetAnalyzes.ContainsNegativeInvestmens))
            //    {
            //        releaseReason = IsTimeToRelease(DataSet.GetDayNum(day));
            //        if (releaseReason != ActionReason.NoReason)
            //        {
            //            IsEndOfInvestment = true;
            //            Action = ActionType.Released;
            //            ActionReason = releaseReason;
            //            return;
            //        }

            //        Action = ActionType.Continued;
            //        ActionReason = ActionReason.SamePredictions;
            //        CountDay = DataSet.GetDayNum(day);
            //        PredictedChange = dataSetAnalyzes.Keys.First();
            //        Analyze = dataSetAnalyzes[PredictedChange];
            //        return;
            //    }
            //}

            //releaseReason = IsTimeToRelease(DataSet.GetDayNum(day));
            //if (releaseReason != ActionReason.NoReason)
            //{
            //    IsEndOfInvestment = true;
            //    Action = ActionType.Released;
            //    ActionReason = releaseReason;
            //    return;
            //}

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
                return balance - GetInvestmentMoney(DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Buy);
            }
            else
            {
                return balance + GetInvestmentMoney(DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], BuySell.Sell);
            }
        }

        public void UpdateRealMoneyOnInvestment(DateTime day, ref double realMoney)
        {
            realMoney -= SimSettings.SafesForStockRate * InvestedMoney;
        }

        public void UpdateRealMoneyOnRelease(DateTime day, ref double realMoney)
        {
            realMoney += SimSettings.SafesForStockRate * InvestedMoney + Profit;
            RealMoney = realMoney;
        }

        public double GetReleaseMoney(DateTime day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            return GetInvestmentMoney(DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
        }

        public double GetPriceFromRatio(DateTime day, double ratio)
        {
            double price;     
            if (InvestmentType == BuySell.Buy)
            {
                price = (ratio + SimSettings.BuySellPenalty * 2) * InvestedPrice + InvestedPrice;
            }
            else
            {
                price =  InvestedPrice - (ratio + SimSettings.BuySellPenalty * 2) * InvestedPrice;
            }

            return (price - DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open]) / DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open];
        }

        public double GetMaxDailyLoosePercentage(DateTime day)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return (DataSet.GetData(DataSet.GetDayNum(day), DataSet.DataColumns.Low) - InvestedPrice) / InvestedPrice - SimSettings.BuySellPenalty * 2;
            }
            else
            {
                return (InvestedPrice - DataSet.GetData(DataSet.GetDayNum(day), DataSet.DataColumns.High)) / InvestedPrice - SimSettings.BuySellPenalty * 2;
            }
        }

        public double GetOpenLoosePercentage(DateTime day)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return (DataSet.GetData(DataSet.GetDayNum(day), DataSet.DataColumns.Open) - InvestedPrice) / InvestedPrice - SimSettings.BuySellPenalty * 2;
            }
            else
            {
                return (InvestedPrice - DataSet.GetData(DataSet.GetDayNum(day), DataSet.DataColumns.Open)) / InvestedPrice - SimSettings.BuySellPenalty * 2;
            }
        }

        public double GetMaxDailyProfitPercentage(DateTime day)
        {
            if (InvestmentType == BuySell.Buy)
            {
                return (DataSet.GetData(DataSet.GetDayNum(day), DataSet.DataColumns.High) - InvestedPrice) / InvestedPrice - SimSettings.BuySellPenalty * 2;
            }
            else
            {
                return (InvestedPrice - DataSet.GetData(DataSet.GetDayNum(day), DataSet.DataColumns.Low)) / InvestedPrice - SimSettings.BuySellPenalty * 2;
            }
        }

        public double GetProfit(DateTime day)
        {
            return (InvestmentType == BuySell.Buy) ? GetReleaseMoney(day) - InvestedMoney : InvestedMoney - GetReleaseMoney(day);
        }

        private double GetProfit(double profitPercentage)
        {
            return InvestedMoney * profitPercentage;
        }

        public double GetInvestmentValue(DateTime day)
        {
            return SimSettings.SafesForStockRate * InvestedMoney + GetProfit(day);
        }

        public double GetInvestmentValue(DateTime day, int offset)
        {
            DateTime offsetDay = DataSet.GetDate(DataSet.GetDayNum(day) + offset);
            return SimSettings.SafesForStockRate * InvestedMoney + GetProfit(offsetDay);
        }

        public double Release(DateTime day, ref double totalValue, double totalProfit, double stockTotalProfit)
        {
            Profit = GetProfit(day);
            InvestmentValue = GetInvestmentValue(day);
            totalProfit += Profit;
            TotalValue = totalValue;
            TotalProfit = totalProfit;
            StockTotalProfit = stockTotalProfit + Profit;
            ReleaseDay = DataSet.GetDayNum(day);

            return StockTotalProfit;
        }

        public double Release(DateTime day, double totalValue, double totalProfit, double stockTotalProfit)
        {
            Profit = GetProfit(day);
            InvestmentValue = GetInvestmentValue(day);
            TotalValue = totalValue;
            TotalProfit = totalProfit;
            StockTotalProfit = stockTotalProfit + Profit;
            ReleaseDay = DataSet.GetDayNum(day);

            return StockTotalProfit;
        }

        public double LimitRelease(DateTime day, double totalValue, double totalProfit, double stockTotalProfit, double limit)
        {
            Profit = GetProfit(limit);
            InvestmentValue = SimSettings.SafesForStockRate * InvestedMoney + Profit;
            TotalValue = totalValue;
            TotalProfit = totalProfit;
            StockTotalProfit = stockTotalProfit + Profit;
            ReleaseDay = DataSet.GetDayNum(day);

            return StockTotalProfit;
        }

        #endregion

        #region Private Methods

        private ActionReason IsTimeToRelease(int day)
        {
            BuySell releaseAction = (InvestmentType == BuySell.Buy) ? BuySell.Sell : BuySell.Buy;
            double releaseMoney = GetInvestmentMoney(DataSet.GetDayData(day)[(int)DataSet.DataColumns.Open], releaseAction);
            double profitRatio = Profit / InvestedMoney;
            if (profitRatio > PredictionsSimulator.MinProfitRatio)
            {
                return ActionReason.GoodProfit;
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
