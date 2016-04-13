﻿using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class AnalyzerSimulator
    {
        #region Members

        List<PredictionRecord> m_PredictionRecords = new List<PredictionRecord>();

        private DateTime m_SimulationDate;

        private double m_MaxAccountBalance = 0.0;

        private double m_MinAccountBalance = 0.0;

        #endregion

        #region Properties

        public Dictionary<string, DataSet> DataSets { get; set; }
        public Dictionary<string, DataPredictions> DataPredictions { get; set; }

        public double AccountBallance { get; set; }

        public double TotalProfit { get; set; }

        public List<Investment> Investments { get; set; }

        static public double EffectivePredictionResult { get; set; }
        
        static public double MinProfitRatio { get; set; }
        
        static public int MaxInvestmentsPerStock { get; set; }
        
        static public int MaxNumOfInvestments { get; set; }

        static public double MaxLooseRatio { get; set; }

        static public string WorkingDirectory { get; set; }

        #endregion

        #region Constructors

        public AnalyzerSimulator(List<string> stocksFiles, string workingDirectory)
        {
            WorkingDirectory = workingDirectory;
            string dataSetsFolder = workingDirectory + DSSettings.DataSetsDir;
            string predictionsDir = workingDirectory + DSSettings.PredictionDir;
            EffectivePredictionResult = 0.9;
            DataSets = new Dictionary<string, DataSet>();
            DataPredictions = new Dictionary<string, DataPredictions>();

            foreach (string stockFile in stocksFiles)
            {
                DataSet dataSet = new DataSet(dataSetsFolder + stockFile, TestDataAction.LoadOnlyTestData);
                DataPredictions dataPredictions = new DataPredictions(predictionsDir + dataSet.DataSetName + DSSettings.PredictionSuffix + ".csv", dataSet);
                DataSets.Add(dataSet.DataSetName, dataSet);
                DataPredictions.Add(dataSet.DataSetName, dataPredictions);
                m_PredictionRecords.AddRange(dataPredictions.GetBestPredictions(AnalyzerSimulator.EffectivePredictionResult));
            }

            m_PredictionRecords = m_PredictionRecords.OrderByDescending(x => x.PredictionCorrectness).ToList();
            Investments = new List<Investment>();
            AccountBallance = 0.0;
            TotalProfit = 0.0;
        }

        #endregion

        #region Interface

        public void Simulate()
        {
            m_MaxAccountBalance = 0.0;
            m_MinAccountBalance = 0.0;
            Log.AddMessage("Simulating, Investment money: {0}", AccountBallance);


            for (double effectivePredictionResult = 0.9; effectivePredictionResult <= 0.9; effectivePredictionResult += 0.002)
            {
                EffectivePredictionResult = effectivePredictionResult;
                m_PredictionRecords.RemoveAll(x => x.PredictionCorrectness < effectivePredictionResult);
                for (double minProfitRatio = 0.005; minProfitRatio <= 0.005; minProfitRatio += 0.01)
                {
                    MinProfitRatio = minProfitRatio;
                    for (double maxLooseRatio = -0.001; maxLooseRatio >= -0.001; maxLooseRatio -= 0.001)
                    {
                        MaxLooseRatio = maxLooseRatio;
                        for (int maxInvestmentPerStock = 25; maxInvestmentPerStock <= 25; maxInvestmentPerStock++)
                        {
                            MaxInvestmentsPerStock = maxInvestmentPerStock;
                            for (int maxNumOfInvestments = 25; maxNumOfInvestments <= 25; maxNumOfInvestments += 25)
                            {
                                MaxNumOfInvestments = maxNumOfInvestments;
                                m_MaxAccountBalance = 0.0;
                                m_MinAccountBalance = 0.0;
                                AccountBallance = 0.0;
                                TotalProfit = 0.0;

                                SimRecorder simRecorder = new SimRecorder(EffectivePredictionResult, MinProfitRatio, MaxInvestmentsPerStock, MaxNumOfInvestments, MaxLooseRatio);
                                for (int dataSetRow = DSSettings.TestRange; dataSetRow >= 0; dataSetRow--)
                                {
                                    m_SimulationDate = new DateTime((long)DataSets.Values.First().GetDayData(dataSetRow)[0]);
                                    Log.AddMessage("Trade date: {0}", m_SimulationDate.ToShortDateString());
                                    RunSimulationCycle(dataSetRow);
                                    simRecorder.AddRecord(dataSetRow, m_SimulationDate, AccountBallance, TotalProfit);
                                }
                                simRecorder.SaveToFile("iForex", WorkingDirectory + SimSettings.SimulationRecordsDirectory);
                            }
                        }
                    }
                }
            }

            Log.AddMessage("Final ammount of money: {0}", AccountBallance);
            Log.AddMessage("Max account balance = {0}, min account balance = {1}", m_MaxAccountBalance, m_MinAccountBalance);
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


        private void RunSimulationCycle(int day)
        {
            Log.AddMessage("{0}:", m_SimulationDate.ToShortDateString());
            List<PredictionRecord> relevantAnalyzerRecords = GetRelevantPredictions(day);
            DailyAnalyzes dailyAnalyzes = GetPredictionsConclusions(day);
           // Log.AddMessage(GetAnalyzeConclussionsReport(dailyAnalyzes));
            dailyAnalyzes.RemoveBadAnalyzes();

            UpdateInvestments(dailyAnalyzes, day);

            ReleaseInvestments(day);

            CreateNewInvestments(dailyAnalyzes, day);
        }

        private void UpdateInvestments(DailyAnalyzes dailyAnalyzes, int dataSetRow)
        {
            foreach (Investment investment in Investments)
            {
                investment.UpdateInvestment(dailyAnalyzes, dataSetRow);
            }
        }

        private void ReleaseInvestments(int day)
        {
            //for (int i = 0; i < Investments.Count; i++)
            //{
            //    ReleaseInvestment(day, Investments.First());
            //}
            List<Investment> investmentsToRelease = Investments.Where(x => x.IsEndOfInvestment).ToList();
            foreach (Investment investment in investmentsToRelease)
            {
                ReleaseInvestment(day, investment);
            }
        }

        private void ReleaseInvestment(int day, Investment investment)
        {
            AccountBallance = investment.UpdateAccountOnRelease(day, AccountBallance);
            if (AccountBallance > m_MaxAccountBalance)
            {
                m_MaxAccountBalance = AccountBallance;
            }
            else if (AccountBallance < m_MinAccountBalance)
            {
                m_MinAccountBalance = AccountBallance;
            }

            if (investment.GetProfit(day) < -100)
            {

            }
            TotalProfit += investment.GetProfit(day);
            Log.AddMessage("Release investment of {0} with prediction {1}:", investment.DataSet.DataSetName, investment.PredictedChange.ToString());
            Log.AddMessage("AccountBalance {0}, release profit {1}, total profit {2}, correctness {3}, {4} predictions", AccountBallance, investment.GetProfit(day), TotalProfit, investment.Analyze.AverageCorrectness, investment.Analyze.NumOfPredictions);
            Investments.Remove(investment);
        }

        private void AddInvestment(int day, Analyze analyze)
        {
            if (Investments.Where(x => x.DataSet.Equals(analyze.DataSet)).Count() >= MaxInvestmentsPerStock)
            {
                return;
            }

            //if (analyze.NumOfPredictions < 100)
            //{
            //    return;
            //}

            if (analyze.PredictedChange.Range > day)
            {
                return;
            }
            Investment investment = new Investment(analyze, day, AccountBallance);
            AccountBallance = investment.UpdateAccountOnInvestment(day, AccountBallance);
            if (AccountBallance > m_MaxAccountBalance)
            {
                m_MaxAccountBalance = AccountBallance;
            }
            else if (AccountBallance < m_MinAccountBalance)
            {
                m_MinAccountBalance = AccountBallance;
            }
            Log.AddMessage("New investment of {0} with prediction {1}, num of investments {2}:", investment.DataSet.DataSetName, investment.PredictedChange.ToString(), Investments.Count + 1);
            Log.AddMessage("Account balance {0}, bought {1} shares, price {2}", AccountBallance, investment.Ammount, investment.InvestedPrice);
            Investments.Add(investment);
        }

        private void CreateNewInvestments(DailyAnalyzes dailyAnalyzes, int day)
        {
            if (day == 0)
            {
                return;
            }

            //List<Analyze> analyzes = new List<Analyze>();
            //foreach (DataSetAnalyzes dataSetAnalyze in dailyAnalyzes.Values)
            //{
            //    analyzes.AddRange(dataSetAnalyze.Values);
            //}

            //foreach (Analyze analyze in analyzes.OrderByDescending(x => x.AverageCorrectness))
            //{
            //    if (Investments.Count < MaxNumOfInvestments)
            //    {
            //        AddInvestment(day, analyze);
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
            foreach (DataSetAnalyzes dataSetAnalyze in dailyAnalyzes.Values)
            {
                foreach (Analyze analyze in dataSetAnalyze.Values.OrderByDescending(x => x.AverageCorrectness))
                {
                    if (Investments.Count < MaxNumOfInvestments)
                    {
                        AddInvestment(day, analyze);
                    }
                }
            }
        }

        private CombinationItem GetRandomPredictedChange()
        {
            Random random = new Random();
            return DSSettings.PredictionItems[random.Next(DSSettings.PredictionItems.Count - 1)];
        }

        private DailyAnalyzes GetPredictionsConclusions(int dataSetRow)
        {
            DailyAnalyzes conclusions = new DailyAnalyzes();
            List<PredictionRecord> relevantPredictions = GetRelevantPredictions(dataSetRow);

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
                report += string.Format("{0} predictions analyze:" + Environment.NewLine, dataset.DataSetName);
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

        private List<PredictionRecord> GetRelevantPredictions(int dataSetRow)
        {
            List<PredictionRecord> fitAnalyzerRecords = new List<PredictionRecord>();
            foreach (PredictionRecord predictionRecord in m_PredictionRecords)
            {
                if (IsAnalyzeFits(dataSetRow, predictionRecord))
                {
                    fitAnalyzerRecords.Add(predictionRecord);
                }
            }
            //for (int i = 0; i < 10 && i < fitAnalyzerRecords.Count; i++)
            //{
            //    //Log.AddMessage("Share {0} should change {1} with probability of {2}", fitAnalyzerRecords[i].DataSetName, fitAnalyzerRecords[i].PredictedChange, fitAnalyzerRecords[i].PredictionCorrectness);
            //}

            return fitAnalyzerRecords;
        }

        private bool IsAnalyzeFits(int dataSetRow, PredictionRecord predictionRecord)
        {
            foreach (CombinationItem combinationItem in predictionRecord.Combination)
            {
                if (!DataPredictions[predictionRecord.DataSet.DataSetName].IsContainsPrediction(combinationItem, dataSetRow, -DSSettings.PredictionErrorRange, DSSettings.PredictionErrorRange))
                {
                    return false;
                }
            }

            return true;
        }



        #endregion
    }
}
