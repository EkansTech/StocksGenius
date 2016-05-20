﻿using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class PredictionsSimulator
    {
        #region Members

        Dictionary<DataSet, List<PredictionRecord>> m_PredictionRecords = new Dictionary<DataSet, List<PredictionRecord>>();

        private static DateTime m_SimulationDate;

        private double m_MaxTotalValue = 0.0;

        private double m_MinTotalValue = 0.0;

        private double m_MaxTotalValueLoose = 0.0;

        InvestmentAnalyzis m_InvestmentAnalyzis;

        private Dictionary<DateTime, DailyAnalyzes> m_DailyAnalyzes = new Dictionary<DateTime, DailyAnalyzes>();

        private AnalyzesSummary m_AnalyzesSummary;

        private int m_SimulationRun = 0;

        public int m_TotalNumOfInvestments = 0;

        public int m_NumOfGoodInvestments = 0;


        private Dictionary<Investment, int> m_SafeModeInvestments = new Dictionary<Investment, int>();

        #endregion

        #region Properties

        public Dictionary<string, DataSet> DataSets { get; set; }

        public Dictionary<string, DataSet> PriceDataSets { get; set; }

        public Dictionary<string, DataPredictions> DataPredictions { get; set; }

        public Dictionary<string, double> StocksTotalProfit { get; set; }

        internal List<Investment> Investments { get; set; }

        static public byte MinPredictedRange { get; set; }

        static public byte MaxPredictedRange { get; set; }

        static public double EffectivePredictionResult { get; set; }

        static public double MinProfitRatio { get; set; }
        
        static public int MaxInvestmentsPerStock { get; set; }
        
        static public double MaxLooseRatio { get; set; }

        static public string WorkingDirectory { get; set; }

        public static double PredictionErrorRange { get; set; }

        public static int MinCombinationItemsNum { get; set; }

        public static int MaxCombinationItemsNum { get; set; }

        public static int SafeDaysNum { get; set; }

        public static int MaxNumOfInvestments { get; set; }

        static public int MaxInvestmentsLive = 1;

        private double m_RealMoney = 2500;

        public double RealMoney
        {
            get { return m_RealMoney; }
            set { m_RealMoney = value; }
        }

        public double InvestmentsValue { get { return Investments.Sum(x => x.GetInvestmentValue(m_SimulationDate)); } }
        public double TotalValue { get { return m_RealMoney + InvestmentsValue; } }

        public int NumOfInvestments { get { return Investments.Count(); } }

        private static int SimDay;

        #endregion

        #region Constructors

        public PredictionsSimulator(DataSetsMetaData metaData, string workingDirectory)
        {
            WorkingDirectory = workingDirectory;
            string dataSetsFolder = workingDirectory + DSSettings.DataSetsDir;
            string priceDataSetsFolder = workingDirectory + DSSettings.PriceDataSetsDir;
            string predictionsDir = workingDirectory + DSSettings.PredictionDir;
            EffectivePredictionResult = DSSettings.EffectivePredictionResult;
            DataSets = new Dictionary<string, DataSet>();
            PriceDataSets = new Dictionary<string, DataSet>();
            DataPredictions = new Dictionary<string, DataPredictions>();
            StocksTotalProfit = new Dictionary<string, double>();

            foreach (string dataSetCode in metaData.Keys)
            {
                DataSet dataSet = new DataSet(dataSetCode, metaData[dataSetCode].DataSetFilePath, TestDataAction.LoadOnlyTestData);
                DataPredictions dataPredictions = new DataPredictions(metaData[dataSetCode].SimDataPredictionsFilePath, dataSet);
                DataSets.Add(dataSet.DataSetCode, dataSet);
                DataPredictions.Add(dataSet.DataSetCode, dataPredictions);
                m_PredictionRecords.Add(dataSet, dataPredictions.GetBestPredictions(PredictionsSimulator.EffectivePredictionResult));

                DataSet priceDataSet = new DataSet(dataSetCode, metaData[dataSetCode].PriceDataSetFilePath, TestDataAction.LoadOnlyTestData);
                PriceDataSets.Add(priceDataSet.DataSetCode, priceDataSet);
                StocksTotalProfit.Add(dataSet.DataSetCode, 0.0);
            }

            Investments = new List<Investment>();
        }

        #endregion

        #region Interface

        public void Simulate()
        {
            m_MaxTotalValue = 0.0;
            m_MinTotalValue = 0.0;
            Log.AddMessage("Simulating, Investment money: {0}", m_RealMoney);

            for (int minCombinationItemsNum = 1; minCombinationItemsNum <= 1; minCombinationItemsNum += 1)
            {
                MinCombinationItemsNum = minCombinationItemsNum;
                for (int maxCombinationItemsNum = 20; maxCombinationItemsNum <= 20; maxCombinationItemsNum += 1)
                {
                    MaxCombinationItemsNum = maxCombinationItemsNum;
                    for (double predictionErrorRange = 0.01; predictionErrorRange <= 0.01; predictionErrorRange += 0.1)
                    {
                        PredictionErrorRange = predictionErrorRange;
                        for (double effectivePredictionResult = 0.75; effectivePredictionResult <= 0.85; effectivePredictionResult += 0.01)
                        {
                            EffectivePredictionResult = effectivePredictionResult;
                            m_DailyAnalyzes.Clear();
                            for (byte minPredictedRange = 1; minPredictedRange <= 1; minPredictedRange += 2)
                            {
                                MinPredictedRange = minPredictedRange;
                                for (byte maxPredictedRange = 12; maxPredictedRange <= 12; maxPredictedRange += 2)
                                {
                                    MaxPredictedRange = maxPredictedRange;
                                    for (double minProfitRatio = 0.25; minProfitRatio <= 0.25; minProfitRatio += 0.01)
                                    {
                                        MinProfitRatio = minProfitRatio;
                                        for (double maxLooseRatio = -0.2; maxLooseRatio >= -0.2; maxLooseRatio -= 0.01)
                                        {
                                            MaxLooseRatio = maxLooseRatio;
                                            for (int maxInvestmentPerStock = 1; maxInvestmentPerStock <= 1; maxInvestmentPerStock++)
                                            {
                                                for (int safeDaysNum = 5; safeDaysNum <= 5; safeDaysNum++)
                                                {
                                                    for (int maxNumOfInvestments = 20; maxNumOfInvestments <= 20; maxNumOfInvestments++)
                                                    {
                                                        MaxNumOfInvestments = maxNumOfInvestments;
                                                        SafeDaysNum = safeDaysNum;
                                                        MaxInvestmentsPerStock = maxInvestmentPerStock;
                                                        m_MaxTotalValue = 0.0;
                                                        m_MinTotalValue = 0.0;
                                                        m_MaxTotalValueLoose = 0.0;
                                                        Investment.Reset();
                                                        m_RealMoney = SimSettings.RealMoneyStartValue;
                                                        m_TotalNumOfInvestments = 0;
                                                        m_NumOfGoodInvestments = 0;
                                                        m_SafeModeInvestments.Clear();

                                                        foreach (string dataSetName in DataSets.Keys)
                                                        {
                                                            StocksTotalProfit[dataSetName] = 0.0;
                                                        }

                                                        m_InvestmentAnalyzis = new InvestmentAnalyzis(WorkingDirectory, m_SimulationRun);
                                                        m_AnalyzesSummary = new AnalyzesSummary(WorkingDirectory, m_SimulationRun);

                                                        SimRecorder simRecorder = new SimRecorder(EffectivePredictionResult, MinProfitRatio, MaxInvestmentsPerStock, MaxLooseRatio, MinPredictedRange, 
                                                            MaxPredictedRange, m_SimulationRun, PredictionErrorRange, MinCombinationItemsNum, MaxCombinationItemsNum, SafeDaysNum, MaxNumOfInvestments);
                                                        for (int dataSetRow = DSSettings.TestRange; dataSetRow >= 0; dataSetRow--)
                                                        {
                                                            SimDay = dataSetRow;
                                                            m_SimulationDate = new DateTime((long)DataSets.Values.First().GetDayData(dataSetRow)[0]);
                                                            Log.AddMessage("Trade date: {0}", m_SimulationDate.ToShortDateString());
                                                            RunSimulationCycle(m_SimulationDate);
                                                            Console.WriteLine("Num of investments: {0}, Real Money: {1}, Investments value: {2}, Total value {3}", NumOfInvestments, m_RealMoney, InvestmentsValue, TotalValue);
                                                            simRecorder.AddRecord(dataSetRow, m_SimulationDate, m_RealMoney, TotalValue, NumOfInvestments);
                                                        }
                                                        simRecorder.SaveToFile("iForex", WorkingDirectory + SimSettings.SimulationRecordsDirectory, m_MaxTotalValue, m_MinTotalValue, m_TotalNumOfInvestments, m_MaxTotalValueLoose, m_NumOfGoodInvestments);
                                                        m_InvestmentAnalyzis.SaveToFile();
                                                        m_AnalyzesSummary.SaveToFile();
                                                        m_SimulationRun++;
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

        public void TestAnalyzeResults(string testFolderPath)
        {
            List<DataPredictions> testDataAnalyzers = new List<DataPredictions>();
            foreach (DataPredictions dataAnalyzer in DataPredictions.Values)
            {
                DataPredictions testDataAnalyzer = dataAnalyzer.PredictionGPUTest();
                testDataAnalyzers.Add(testDataAnalyzer);
                testDataAnalyzer.SaveDataToFile(testFolderPath);
            }
            CompareTestResults(DataPredictions.Values.ToList(), testDataAnalyzers);
        }

        static public void CompareTestResults(List<DataPredictions> dataAnalyzers, List<DataPredictions> testDataAnalyzers)
        {
            double[] totalNumOfPreditions = new double[dataAnalyzers[0].NumOfDataColumns];
            double[] totalSum = new double[dataAnalyzers[0].NumOfDataColumns];
            double[] totalTestSum = new double[dataAnalyzers[0].NumOfDataColumns];

            for (int i = 0; i < dataAnalyzers.Count; i++)
            {
                DataPredictions dataAnalyzer = dataAnalyzers[i];
                DataPredictions testDataAnalyzer = testDataAnalyzers[i];
                double[] numOfPreditions = new double[dataAnalyzer.NumOfDataColumns];
                double[] sum = new double[dataAnalyzer.NumOfDataColumns];
                double[] testSum = new double[dataAnalyzer.NumOfDataColumns];
                foreach (ulong combination in dataAnalyzer.Keys)
                {
                    for (int j = 0; j < dataAnalyzer.NumOfDataColumns; j++)
                    {
                        if (dataAnalyzer[combination][j] >= EffectivePredictionResult)
                        {
                            sum[j] += dataAnalyzer[combination][j];
                            testSum[j] += testDataAnalyzer[combination][j];
                            numOfPreditions[j]++;
                        }
                    }
                }

                for (int j = 0; j < dataAnalyzer.NumOfDataColumns; j++)
                {
                    totalNumOfPreditions[j] += numOfPreditions[j];
                    totalSum[j] += sum[j];
                    totalTestSum[j] += testSum[j];

                    if (numOfPreditions[j] > 0)
                    {
                        Console.WriteLine("Prediction {0}, predicted average {1}, tested average {2}, num of predictions {3}",
                            DSSettings.PredictionItems[j].ToString(), sum[j] / numOfPreditions[j], testSum[j] / numOfPreditions[j], numOfPreditions[j]);
                    }
                }
            }

            Console.WriteLine("Total results:");
            double finalNumOfPreditions = 0.0;
            double finalSum = 0.0;
            double finalTestSum = 0.0;

            for (int j = 0; j < dataAnalyzers[0].NumOfDataColumns; j++)
            {
                finalNumOfPreditions += totalNumOfPreditions[j];
                finalSum += totalSum[j];
                finalTestSum += totalTestSum[j];
                if (totalNumOfPreditions[j] > 0)
                {
                    Console.WriteLine("Prediction {0}, predicted average {1}, tested average {2}, num of predictions {3}",
                        DSSettings.PredictionItems[j].ToString(), totalSum[j] / totalNumOfPreditions[j], totalTestSum[j] / totalNumOfPreditions[j], totalNumOfPreditions[j]);
                }
            }

            Console.WriteLine("Final accuracy: predicted average {0}, tested average {1}, num of predictions {2}",
                finalSum / finalNumOfPreditions, finalTestSum / finalNumOfPreditions, finalNumOfPreditions);
        }

        #endregion

        #region Private Methods

        private void CompareTestResults(DataPredictions dataAnalyzer, DataPredictions testDataAnalyzer)
        {
            double predictionLimit = 0.85;
            double[] numOfPreditions = new double[dataAnalyzer.NumOfDataColumns];
            double[] sum = new double[dataAnalyzer.NumOfDataColumns];
            double[] testSum = new double[dataAnalyzer.NumOfDataColumns];
            foreach (ulong combination in dataAnalyzer.Keys)
            {
                for (int i = 0; i < dataAnalyzer.NumOfDataColumns; i++)
                {
                    if (dataAnalyzer[combination][i] >= predictionLimit)
                    {
                        sum[i] += dataAnalyzer[combination][i];
                        testSum[i] += testDataAnalyzer[combination][i];
                        numOfPreditions[i]++;
                    }
                }
            }

            for (int i = 0; i < dataAnalyzer.NumOfDataColumns; i++)
            {
                if (numOfPreditions[i] > 0)
                {
                    Console.WriteLine("Prediction {0}, predicted average {1}, tested average {2}, num of predictions {3}",
                        DSSettings.PredictionItems[i].ToString(), sum[i] / numOfPreditions[i], testSum[i] / numOfPreditions[i], numOfPreditions[i]);
                }
            }
        }


        private void RunSimulationCycle(DateTime day)
        {
            Log.AddMessage("{0}:", m_SimulationDate.ToShortDateString());
            DailyAnalyzes dailyAnalyzes;
            if (m_DailyAnalyzes.ContainsKey(day))
            {
                dailyAnalyzes = m_DailyAnalyzes[day];
            }
            else
            {
                dailyAnalyzes = GetPredictionsConclusions(day);
                m_DailyAnalyzes.Add(day, dailyAnalyzes);
            }

            m_AnalyzesSummary.Add(m_SimulationRun, day, dailyAnalyzes);
           // Log.AddMessage(GetAnalyzeConclussionsReport(dailyAnalyzes));
            dailyAnalyzes.RemoveBadAnalyzes();

            UpdateSafeMode(dailyAnalyzes, day);

            UpdateInvestments(dailyAnalyzes, day);

            ReleaseInvestments(day);

            CreateNewInvestments(dailyAnalyzes, day);
        }

        private void UpdateSafeMode(DailyAnalyzes dailyAnalyzes, DateTime day)
        {
            List<Investment> investments = m_SafeModeInvestments.Keys.ToList();
            foreach (Investment investment in investments)
            {
                if ((investment.GetInvestmentValue(day) - investment.GetInvestmentValue(day, 1)) / investment.GetInvestmentValue(day, 1) > 0)
                {
                    m_SafeModeInvestments.Remove(investment);
                }
                //else if (dailyAnalyzes.ContainsKey(investment.DataSet) && dailyAnalyzes[investment.DataSet].Where(x => DSSettings.OppositeDataItems[x.Key.DataItem] == investment.PredictedChange.DataItem).Count() > 0)
                //{
                //    m_SafeModeInvestments.Remove(investment);
                //}
                else if (m_SafeModeInvestments[investment] - investment.DataSet.GetDayNum(day) >= SafeDaysNum)
                {
                    m_SafeModeInvestments.Remove(investment);
                }
                //if (dailyAnalyzes.ContainsKey(investment.DataSet) && dailyAnalyzes[investment.DataSet].Where(x => DSSettings.OppositeDataItems[x.Key.DataItem] == investment.PredictedChange.DataItem).Count() > 0)
                //{
                //    m_SafeModeInvestments.Remove(investment);
                //}
                //else if (dailyAnalyzes.ContainsKey(investment.DataSet) && dailyAnalyzes[investment.DataSet].Where(x => x.Key.DataItem == investment.PredictedChange.DataItem).Count() > 0)
                //{
                //    m_SafeModeInvestments[investment] = day;
                //} 
                //else if (!dailyAnalyzes.ContainsKey(investment.DataSet) && m_SafeModeInvestments[investment] - day >= 5)
                //{
                //    m_SafeModeInvestments.Remove(investment);
                //}
            }

            if ((Investments.Sum(x => x.GetInvestmentValue(day)) - Investments.Sum(x => x.GetInvestmentValue(day, 1))) / Investments.Sum(x => x.GetInvestmentValue(day, 1)) < MaxLooseRatio)
            {
                if (TotalValue > 1000)
                {

                    investments = Investments.ToList();
                    foreach (Investment investment in investments)
                    {
                        m_SafeModeInvestments.Add(investment, investment.DataSet.GetDayNum(day));

                        investment.UpdateInvestment(day, TotalValue, StocksTotalProfit[investment.DataSet.DataSetCode]);
                        investment.Action = ActionType.Released;
                        investment.ActionReason = ActionReason.SafeMode;
                        investment.ReleaseDay = investment.DataSet.GetDayNum(day);
                        m_InvestmentAnalyzis.Add(investment, day);
                        ReleaseInvestment(day, investment);
                    }
                }
            }
            //investments = Investments.ToList();
            //foreach (Investment investment in investments)
            //{
            //    if ((investment.GetInvestmentValue(day) - investment.GetInvestmentValue(day + 1)) / investment.GetInvestmentValue(day + 1) < MaxLooseRatio)
            //    {
            //        if (m_SafeModeInvestments.Where(x => x.Key.DataSet.DataSetCode.Equals(investment.DataSet.DataSetCode)).Count() == 0)
            //        {
            //            m_SafeModeInvestments.Add(investment, day);

            //            foreach (Investment releaseInvestment in investments.Where(x => x.DataSet.DataSetCode.Equals(investment.DataSet.DataSetCode)))
            //            {
            //                releaseInvestment.UpdateInvestment(day, TotalValue, StocksTotalProfit[investment.DataSet.DataSetCode]);
            //                releaseInvestment.Action = ActionType.Released;
            //                releaseInvestment.ActionReason = ActionReason.SafeMode;
            //                m_InvestmentAnalyzis.Add(releaseInvestment, day);
            //                ReleaseInvestment(day, releaseInvestment);
            //            }
            //        }
            //    }
            //}

            foreach (DataSet dataSet in m_SafeModeInvestments.Keys.Select(x => x.DataSet))
            {
                if (dailyAnalyzes.ContainsKey(dataSet))
                {
                    dailyAnalyzes.Remove(dataSet);
                }
            }
        }

        private void UpdateInvestments(DailyAnalyzes dailyAnalyzes, DateTime day)
        {
            foreach (Investment investment in Investments.OrderBy(x => x.ID))
            {
                investment.UpdateInvestment(dailyAnalyzes, day, TotalValue, RealMoney, StocksTotalProfit[investment.DataSet.DataSetCode]);
                m_InvestmentAnalyzis.Add(investment, day);
            }
        }

        private void ReleaseInvestments(DateTime day)
        {
            //for (int i = 0; i < Investments.Count; i++)
            //{
            //    ReleaseInvestment(day, Investments.First());
            //}
            List<Investment> investmentsToRelease = Investments.Where(x => x.IsEndOfInvestment).ToList();
            foreach (Investment investment in investmentsToRelease.OrderBy(x => x.ID))
            {
                ReleaseInvestment(day, investment);
            }
        }

        private void ReleaseInvestment(DateTime day, Investment investment)
        {
            //if (investment.Profit > SimSettings.MaxProfit || investment.Profit < SimSettings.MaxLoose)
            //{
            //    Log.AddMessage("Investment {0} profit {1} is out of bounds, dismissed", investment.PredictedChange.ToString(), investment.Profit);
            //    return;
            //}

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

            //if (analyze.NumOfPredictions < 100)
            //{
            //    return;
            //}

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

        private void CreateNewInvestments(DailyAnalyzes dailyAnalyzes, DateTime day)
        {
            if (DataSets.Values.First().GetDayNum(day) == 0)
            {
                return;
            }

            List<Analyze> analyzes = new List<Analyze>();
            foreach (DataSet dataSet in dailyAnalyzes.Keys)
            {
                analyzes.AddRange(dailyAnalyzes[dataSet].Values);
            }

            var orderAnalyzes = analyzes.OrderByDescending(x => x.AverageCorrectness);//OrderBy(x => x, new AnalyzeComparer());//

            foreach (Analyze analyze in orderAnalyzes)
            {
                if (NumOfInvestments >= MaxNumOfInvestments)
                {
                    return;
                }
                    //double average = CalculateAverage(analyze.DataSet, day, analyze.PredictedChange.Range, DataSet.DataColumns.Open);
                    //if ((analyze.IsPositiveInvestment && analyze.DataSet.GetData(day, DataSet.DataColumns.Open) > average * 1.01)
                    //    ||(analyze.IsNegativeInvestment && analyze.DataSet.GetData(day, DataSet.DataColumns.Open) < average ))
                    //{
                    //    continue;
                    //}
                    double accountValue = m_RealMoney + Investments.Sum(x => x.Profit);
                if (((SimSettings.SafesForStockRate * SimSettings.InvestmentPerStock) < accountValue) && analyze.PredictedChange.Range <= MaxPredictedRange && analyze.PredictedChange.Range >= MinPredictedRange)
                {
                    if (Investments.Where(x => x.DataSet.DataSetCode.Equals(analyze.DataSet.DataSetCode)).Count() < MaxInvestmentsPerStock)
                    {
                        AddInvestment(day, analyze, 0.0);
                    }
                }
            }

            //for (double percentage = 0.1; percentage <= 0.1; percentage += 0.01)
            //{
            //    foreach (Analyze analyze in orderAnalyzes)
            //    {
            //        if (NumOfInvestments >= MaxNumOfInvestments)
            //        {
            //            return;
            //        }
            //        if (Investments.Where(x => x.DataSet.DataSetCode.Equals(analyze.DataSet.DataSetCode)).Count() >= MaxInvestmentsPerStock * 2)
            //        {
            //            continue;
            //        }
            //        double accountValue = m_RealMoney + Investments.Sum(x => x.Profit);
            //        if (((SimSettings.SafesForStockRate * SimSettings.InvestmentPerStock) < accountValue) && analyze.PredictedChange.Range <= MaxPredictedRange && analyze.PredictedChange.Range >= MinPredictedRange)
            //        {
            //            if (analyze.IsPositiveInvestment && ((analyze.DataSet.GetData(day, DataSet.DataColumns.Open) - analyze.DataSet.GetData(day, DataSet.DataColumns.Low)) / analyze.DataSet.GetData(day, DataSet.DataColumns.Open)) >= percentage)
            //            {
            //                AddInvestment(day, analyze, -percentage);
            //            }
            //            else if (!analyze.IsPositiveInvestment && ((analyze.DataSet.GetData(day, DataSet.DataColumns.High) - analyze.DataSet.GetData(day, DataSet.DataColumns.Open)) / analyze.DataSet.GetData(day, DataSet.DataColumns.Open)) >= percentage)
            //            {
            //                AddInvestment(day, analyze, percentage);
            //            }
            //        }
            //        //double closeOpen;
            //        //if ((analyze.IsPositiveInvestment && (analyze.DataSet.GetData(day, DataSet.DataColumns.Open) > analyze.DataSet.GetData(day, DataSet.DataColumns.Close)))
            //        //    || (analyze.IsNegativeInvestment && (analyze.DataSet.GetData(day, DataSet.DataColumns.Open) < analyze.DataSet.GetData(day, DataSet.DataColumns.Close))))
            //        //{
            //        //    closeOpen = 0.0;
            //        //}
            //        //else
            //        //{
            //        //    closeOpen = (analyze.DataSet.GetData(day, DataSet.DataColumns.Close) - analyze.DataSet.GetData(day, DataSet.DataColumns.Open)) / analyze.DataSet.GetData(day, DataSet.DataColumns.Open);
            //        //}
            //        //if (((SimSettings.SafesForStockRate * SimSettings.InvestmentPerStock) < accountValue) && analyze.PredictedChange.Range <= MaxPredictedRange && analyze.PredictedChange.Range >= MinPredictedRange)
            //        //{
            //        //    if (!analyze.IsPositiveInvestment && ((analyze.DataSet.GetData(day, DataSet.DataColumns.Open) - analyze.DataSet.GetData(day, DataSet.DataColumns.Low)) / analyze.DataSet.GetData(day, DataSet.DataColumns.Open)) >= percentage)
            //        //    {
            //        //        AddInvestment(day, analyze, closeOpen);
            //        //    }
            //        //    else if (analyze.IsPositiveInvestment && ((analyze.DataSet.GetData(day, DataSet.DataColumns.High) - analyze.DataSet.GetData(day, DataSet.DataColumns.Open)) / analyze.DataSet.GetData(day, DataSet.DataColumns.Open)) >= percentage)
            //        //    {
            //        //        AddInvestment(day, analyze, closeOpen);
            //        //    }
            //        //}
            //    }
            //}

            //return;

            //foreach (DataSet dataset in DataSets.Values)
            //{
            //    if (Investments.Count < MaxNumOfInvestments)
            //    {
            //        Analyze analyze = new Analyze(new PredictionRecord() { Combination = null, DataSet = dataset, PredictedChange = GetRandomPredictedChange(), PredictionCorrectness = 0.9 })
            //        {
            //            NumOfPredictions = 200,
            //            AverageCorrectness = 0.9,
            //            DataSet = dataset,
            //        };
            //        AddInvestment(day, analyze);
            //    }

            //}


            //foreach (DataSetAnalyzes dataSetAnalyze in dailyAnalyzes.Values)
            //{
            //    foreach (Analyze analyze in dataSetAnalyze.Values.OrderBy(x => x, new AnalyzeComparer())) //OrderByDescending(x => x.AverageCorrectness)) 
            //    {
            //        if (Investments.Count < MaxNumOfInvestments && analyze.PredictedChange.Range <= MaxPredictedRange)
            //        {
            //            AddInvestment(day, analyze);
            //        }
            //    }
            //}
        }

        private CombinationItem GetRandomPredictedChange()
        {
            Random random = new Random();
            return DSSettings.PredictionItems[random.Next(DSSettings.PredictionItems.Count - 1)];
        }

        private DailyAnalyzes GetPredictionsConclusions(DateTime day)
        {
            DailyAnalyzes conclusions = new DailyAnalyzes();
            List<PredictionRecord> relevantPredictions = GetRelevantPredictions(day);

            foreach (PredictionRecord record in relevantPredictions)
            {
                conclusions.Add(record.DataSet, record.PredictedChange, record);
            }

            return conclusions;
        }

        private string GetAnalyzeConclussionsReport(DailyAnalyzes analyzeConclussions)
        {
            string report = string.Empty;
            foreach (DataSet dataset in analyzeConclussions.Keys)
            {
                report += string.Format("{0} predictions analyze:" + Environment.NewLine, dataset.DataSetCode);
                foreach (CombinationItem predictedChange in analyzeConclussions[dataset].Keys)
                {
                    Analyze conclusion = analyzeConclussions[dataset][predictedChange];
                    report += string.Format("Change of {0}, range {1}, {2} of predictions, accuracy {3}",
                        conclusion.PredictedChange.DataItem, conclusion.PredictedChange.Range, conclusion.NumOfPredictions, conclusion.AverageCorrectness);

                    report += Environment.NewLine;
                }
            }

            return report;
        }

        private List<PredictionRecord> GetRelevantPredictions(DateTime day)
        {
            List<PredictionRecord> fitAnalyzerRecords = new List<PredictionRecord>();
            foreach (DataSet dataSet in m_PredictionRecords.Keys)
            {
                int dayNum = dataSet.GetDayNum(day);
                if (dayNum == -1 || dataSet.GetDate(dayNum) != day)
                {
                    continue;
                }
                foreach (PredictionRecord predictionRecord in m_PredictionRecords[dataSet].Where(x => x.PredictionCorrectness >= EffectivePredictionResult
                && x.Combination.Count >= MinCombinationItemsNum
                && x.Combination.Count <= MaxCombinationItemsNum))
                {
                    //bool[] goodPredictions = 
                    if (IsAnalyzeFits(dayNum, predictionRecord))
                    {
                        fitAnalyzerRecords.Add(predictionRecord);
                    }
                }
            }

            return fitAnalyzerRecords;
        }

        private bool IsAnalyzeFits(int dataSetRow, PredictionRecord predictionRecord)
        {
            foreach (CombinationItem combinationItem in predictionRecord.Combination)
            {
                if (!DataPredictions[predictionRecord.DataSet.DataSetCode].IsContainsPrediction(combinationItem, dataSetRow, PredictionErrorRange, -PredictionErrorRange))
                {
                    return false;
                }
            }

            return true;
        }

        //public bool IsContainsGoodPrediction(DataSet dataSet, CombinationItem combinationItem, int dataRow, double upperErrorBorder, double lowerErrorBorder)
        //{
        //    double currentOpenAverage = CalculateAverage(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Open);
        //    double currentCloseAverage = CalculateAverage(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Close);
        //    double prevOpenAverage = CalculateAverage(dataSet, dataRow + combinationItem.Range, combinationItem.Range, DataSet.DataColumns.Open);
        //    double prevCloseAverage = CalculateAverage(dataSet, dataRow + combinationItem.Range, combinationItem.Range, DataSet.DataColumns.Close);
        //    double currentVolumeAverage = CalculateAverage(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Volume);
        //    double prevVolumeAverage = CalculateAverage(dataSet, dataRow + combinationItem.Range, combinationItem.Range, DataSet.DataColumns.Volume);

        //    if (combinationItem.DataItem == DataItem.OpenChange
        //        && (currentOpenAverage - prevOpenAverage) / prevOpenAverage > upperErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] < currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.CloseChange
        //        && (currentCloseAverage - prevCloseAverage) / prevCloseAverage > upperErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] < currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.VolumeChange
        //        && (currentVolumeAverage - prevVolumeAverage) / prevVolumeAverage > upperErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] < currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.CloseOpenDif
        //        && (currentCloseAverage - currentOpenAverage) / currentOpenAverage > upperErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] < currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.OpenPrevCloseDif
        //        && (currentOpenAverage - prevCloseAverage) / prevCloseAverage > upperErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] < currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.PrevCloseOpenDif
        //        && (prevCloseAverage - prevOpenAverage) / prevOpenAverage > upperErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] < currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeOpenChange
        //        && (currentOpenAverage - prevOpenAverage) / prevOpenAverage < lowerErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] > currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeCloseChange
        //        && (currentCloseAverage - prevCloseAverage) / prevCloseAverage < lowerErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] > currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeVolumeChange
        //        && (currentVolumeAverage - prevVolumeAverage) / prevVolumeAverage < lowerErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] > currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeCloseOpenDif
        //        && (currentCloseAverage - currentOpenAverage) / currentOpenAverage < lowerErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] > currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeOpenPrevCloseDif
        //        && (currentOpenAverage - prevCloseAverage) / prevCloseAverage < lowerErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] > currentOpenAverage)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativePrevCloseOpenDif
        //        && (prevCloseAverage - prevOpenAverage) / prevOpenAverage < lowerErrorBorder
        //        && dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)DataSet.DataColumns.Open] > currentOpenAverage)
        //    { return true; }

        //    return false;
        //}

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
