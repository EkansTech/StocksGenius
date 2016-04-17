using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class LatestSimulator
    {
        #region Members

        List<PredictionRecord> m_PredictionRecords = new List<PredictionRecord>();

        private DateTime m_SimulationDate;

        private double m_MaxAccountBalance = 0.0;

        private double m_MinAccountBalance = 0.0;

        #endregion

        #region Properties

        public Dictionary<string, DataSet> DataSets { get; set; }

        public Dictionary<string, DataSet> PriceDataSets { get; set; }

        public LatestPredictions LatestPredictions { get; set; }

        public double AccountBallance { get; set; }

        public double TotalProfit { get; set; }

        public List<Investment> Investments { get; set; }

        static public double EffectivePredictionResult { get; set; }

        static public double MinProfitRatio { get; set; }

        static public int MaxInvestmentsPerStock { get; set; }

        static public int MaxNumOfInvestments { get; set; }

        static public double MaxLooseRatio { get; set; }

        #endregion

        #region Constructors

        public LatestSimulator(List<string> dataSetPaths, string latestPredictionsFilePath)
        {
            EffectivePredictionResult = 0.9;
            DataSets = new Dictionary<string, DataSet>();

            foreach (string stockFile in dataSetPaths)
            {
                DataSet dataSet = new DataSet(stockFile, TestDataAction.LoadOnlyTestData);
                DataSets.Add(dataSet.DataSetName, dataSet);

                DataSet priceDataSet = new DataSet(stockFile.Replace(DSSettings.DataSetsDir, SimSettings.PriceDataSetsDirectory), TestDataAction.LoadOnlyTestData);
                DataSets.Add(priceDataSet.DataSetName, priceDataSet);
            }
            
            LatestPredictions = new LatestPredictions(DataSets.Values.ToList(), latestPredictionsFilePath);

            m_PredictionRecords = LatestPredictions.GetBestPredictions(LatestSimulator.EffectivePredictionResult).OrderByDescending(x => x.PredictionCorrectness).ToList();
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


            for (double effectivePredictionResult = 0.9; effectivePredictionResult <= 0.9; effectivePredictionResult += 0.002F)
            {
                EffectivePredictionResult = effectivePredictionResult;
                m_PredictionRecords.RemoveAll(x => x.PredictionCorrectness < effectivePredictionResult);
                for (double minProfitRatio = 0.005; minProfitRatio <= 0.005; minProfitRatio += 0.01F)
                {
                    MinProfitRatio = minProfitRatio;
                    for (double maxLooseRatio = -0.001; maxLooseRatio >= -0.001; maxLooseRatio -= 0.001F)
                    {
                        MaxLooseRatio = maxLooseRatio;
                        for (int maxInvestmentPerStock = 1; maxInvestmentPerStock <= 1; maxInvestmentPerStock++)
                        {
                            MaxInvestmentsPerStock = maxInvestmentPerStock;
                            for (int maxNumOfInvestments = 25; maxNumOfInvestments <= 25; maxNumOfInvestments += 25)
                            {
                                MaxNumOfInvestments = maxNumOfInvestments;
                                m_MaxAccountBalance = 0.0;
                                m_MinAccountBalance = 0.0;
                                AccountBallance = 0.0;
                                TotalProfit = 0.0;

                                SimRecorder simRecorder = new SimRecorder(EffectivePredictionResult, MinProfitRatio, MaxInvestmentsPerStock, MaxNumOfInvestments, MaxLooseRatio, 12);
                                for (int dataSetRow = DSSettings.TestRange; dataSetRow >= 0; dataSetRow--)
                                {
                                    m_SimulationDate = new DateTime((long)DataSets.Values.First().GetDayData(dataSetRow)[0]);
                                    Log.AddMessage("Trade date: {0}", m_SimulationDate.ToShortDateString());
                                    RunSimulationCycle(dataSetRow);
                                    simRecorder.AddRecord(dataSetRow, m_SimulationDate, AccountBallance, TotalProfit);
                                }
                                simRecorder.SaveToFile("iForex", "C:\\Ekans\\Stocks\\Quandl\\\\iForexAnalyzerRecords\\");
                            }
                        }
                    }
                }
            }

            Log.AddMessage("Final ammount of money: {0}", AccountBallance);
            Log.AddMessage("Max account balance = {0}, min account balance = {1}", m_MaxAccountBalance, m_MinAccountBalance);
        }

        #endregion

        #region Private Methods

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

            if (analyze.NumOfPredictions < 100)
            {
                return;
            }

            if (analyze.PredictedChange.Range > day)
            {
                return;
            }
            Investment investment = new Investment(PriceDataSets[analyze.DataSet.DataSetName], analyze, day, AccountBallance);
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
            //if (day == 0)
            //{
            //    return;
            //}

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
            foreach (DataSet dataSet in DataSets.Values)
            {
                foreach (PredictionRecord predictionRecord in m_PredictionRecords)
                {
                    if (IsAnalyzeFits(dataSetRow, predictionRecord, dataSet))
                    {
                        PredictionRecord newPredictionRecord = new PredictionRecord(predictionRecord, dataSet);
                        fitAnalyzerRecords.Add(newPredictionRecord);
                    }
                }
            }
            //for (int i = 0; i < 10 && i < fitAnalyzerRecords.Count; i++)
            //{
            //    //Log.AddMessage("Share {0} should change {1} with probability of {2}", fitAnalyzerRecords[i].DataSetName, fitAnalyzerRecords[i].PredictedChange, fitAnalyzerRecords[i].PredictionCorrectness);
            //}

            return fitAnalyzerRecords;
        }

        private bool IsAnalyzeFits(int dataSetRow, PredictionRecord predictionRecord, DataSet dataset)
        {
            foreach (CombinationItem combinationItem in predictionRecord.Combination)
            {
                if (!LatestPredictions.IsContainsPrediction(dataset, combinationItem, dataSetRow, -DSSettings.PredictionErrorRange, DSSettings.PredictionErrorRange))
                {
                    return false;
                }
            }

            return true;
        }



        #endregion
    }
}
