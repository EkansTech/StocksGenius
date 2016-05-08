using StocksData;
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

        private DateTime m_SimulationDate;

        private double m_MaxTotalProfit = 0.0;

        private double m_MinTotalProfit = 0.0;

        private int m_TotalNumOfInvestments = 0;

        InvestmentAnalyzis m_InvestmentAnalyzis;

        private Dictionary<int, DailyAnalyzes> m_DailyAnalyzes = new Dictionary<int, DailyAnalyzes>();

        private AnalyzesSummary m_AnalyzesSummary;

        private int m_SimulationRun = 0;

        #endregion

        #region Properties

        public Dictionary<string, DataSet> DataSets { get; set; }

        public Dictionary<string, DataSet> PriceDataSets { get; set; }

        public Dictionary<string, DataPredictions> DataPredictions { get; set; }

        public double AccountBallance { get; set; }

        private double m_TotalProfit;

        public double TotalProfit
        {
            get { return m_TotalProfit; }
            set { m_TotalProfit = value; }
        }

        public Dictionary<string, double> StocksTotalProfit { get; set; }

        internal List<Investment> Investments { get; set; }

        static public byte MinPredictedRange { get; set; }

        static public byte MaxPredictedRange { get; set; }

        static public double EffectivePredictionResult { get; set; }
        
        static public double MinProfitRatio { get; set; }
        
        static public int MaxInvestmentsPerStock { get; set; }
        
        static public int MaxNumOfInvestments { get; set; }

        static public double MaxLooseRatio { get; set; }

        static public string WorkingDirectory { get; set; }

        public static double PredictionErrorRange { get; set; }

        public static int MinCombinationItemsNum { get; set; }

        public static int MaxCombinationItemsNum { get; set; }

        static public int MaxInvestmentsLive = 7;

        #endregion

        #region Constructors

        public PredictionsSimulator(List<string> stocksFiles, string workingDirectory)
        {
            WorkingDirectory = workingDirectory;
            string dataSetsFolder = workingDirectory + DSSettings.DataSetsDir;
            string priceDataSetsFolder = workingDirectory + DSSettings.PriceDataSetsDirectory;
            string predictionsDir = workingDirectory + DSSettings.PredictionDir;
            EffectivePredictionResult = DSSettings.MinimumRelevantPredictionResult;
            DataSets = new Dictionary<string, DataSet>();
            PriceDataSets = new Dictionary<string, DataSet>();
            DataPredictions = new Dictionary<string, DataPredictions>();
            StocksTotalProfit = new Dictionary<string, double>();

            foreach (string stockFile in stocksFiles)
            {
                DataSet dataSet = new DataSet(dataSetsFolder + stockFile, TestDataAction.LoadOnlyTestData);
                DataPredictions dataPredictions = new DataPredictions(predictionsDir + dataSet.DataSetName + DSSettings.PredictionSuffix + ".csv", dataSet);
                DataSets.Add(dataSet.DataSetName, dataSet);
                DataPredictions.Add(dataSet.DataSetName, dataPredictions);
                m_PredictionRecords.Add(dataSet, dataPredictions.GetBestPredictions(PredictionsSimulator.EffectivePredictionResult));

                DataSet priceDataSet = new DataSet(priceDataSetsFolder + stockFile, TestDataAction.LoadOnlyTestData);
                PriceDataSets.Add(priceDataSet.DataSetName, priceDataSet);
                StocksTotalProfit.Add(dataSet.DataSetName, 0.0);
            }

            Investments = new List<Investment>();
            AccountBallance = 0.0;
            TotalProfit = 0.0;
        }

        #endregion

        #region Interface

        public void Simulate()
        {
            m_MaxTotalProfit = 0.0;
            m_MinTotalProfit = 0.0;
            Log.AddMessage("Simulating, Investment money: {0}", AccountBallance);

            for (int minCombinationItemsNum = 1; minCombinationItemsNum <= 1; minCombinationItemsNum += 1)
            {
                MinCombinationItemsNum = minCombinationItemsNum;
                for (int maxCombinationItemsNum = 20; maxCombinationItemsNum <= 20; maxCombinationItemsNum += 1)
                {
                    MaxCombinationItemsNum = maxCombinationItemsNum;
                    for (double predictionErrorRange = 0.01; predictionErrorRange <= 0.01; predictionErrorRange += 0.1)
                    {
                        PredictionErrorRange = predictionErrorRange;
                        for (double effectivePredictionResult = 0.9; effectivePredictionResult <= 1; effectivePredictionResult += 0.01)
                        {
                            EffectivePredictionResult = effectivePredictionResult;
                            m_DailyAnalyzes.Clear();
                            for (byte minPredictedRange = 1; minPredictedRange <= 1; minPredictedRange += 2)
                            {
                                MinPredictedRange = minPredictedRange;
                                for (byte maxPredictedRange = 12; maxPredictedRange <= 12; maxPredictedRange += 2)
                                {
                                    MaxPredictedRange = maxPredictedRange;
                                    for (double minProfitRatio = 0.3; minProfitRatio <= 0.3; minProfitRatio += 0.01)
                                    {
                                        MinProfitRatio = minProfitRatio;
                                        for (double maxLooseRatio = -0.3; maxLooseRatio >= -0.3; maxLooseRatio -= 0.01)
                                        {
                                            MaxLooseRatio = maxLooseRatio;
                                            for (int maxInvestmentPerStock = 1; maxInvestmentPerStock <= 5; maxInvestmentPerStock++)
                                            {
                                                MaxInvestmentsPerStock = maxInvestmentPerStock;
                                                for (int maxNumOfInvestments = 5; maxNumOfInvestments <= 20; maxNumOfInvestments += 1)
                                                {
                                                    MaxNumOfInvestments = maxNumOfInvestments;
                                                    m_MaxTotalProfit = 0.0;
                                                    m_MinTotalProfit = 0.0;
                                                    AccountBallance = 0.0;
                                                    TotalProfit = 0.0;
                                                    m_TotalNumOfInvestments = 0;
                                                    Investment.Reset();

                                                    foreach (string dataSetName in DataSets.Keys)
                                                    {
                                                        StocksTotalProfit[dataSetName] = 0.0;
                                                    }

                                                    m_InvestmentAnalyzis = new InvestmentAnalyzis(WorkingDirectory, m_SimulationRun);
                                                    m_AnalyzesSummary = new AnalyzesSummary(WorkingDirectory, m_SimulationRun);

                                                    SimRecorder simRecorder = new SimRecorder(EffectivePredictionResult, MinProfitRatio, MaxInvestmentsPerStock, MaxNumOfInvestments, MaxLooseRatio,
                                                        MinPredictedRange, MaxPredictedRange, m_SimulationRun, PredictionErrorRange, MinCombinationItemsNum, MaxCombinationItemsNum);
                                                    for (int dataSetRow = DSSettings.TestRange; dataSetRow >= 0; dataSetRow--)
                                                    {
                                                        m_SimulationDate = new DateTime((long)DataSets.Values.First().GetDayData(dataSetRow)[0]);
                                                        Log.AddMessage("Trade date: {0}", m_SimulationDate.ToShortDateString());
                                                        RunSimulationCycle(dataSetRow);
                                                        simRecorder.AddRecord(dataSetRow, m_SimulationDate, AccountBallance, TotalProfit);
                                                    }
                                                    simRecorder.SaveToFile("iForex", WorkingDirectory + SimSettings.SimulationRecordsDirectory, m_MaxTotalProfit, m_MinTotalProfit, m_TotalNumOfInvestments);
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

            Log.AddMessage("Final ammount of money: {0}", AccountBallance);
            Log.AddMessage("Max total profit = {0}, min total profit = {1}", m_MaxTotalProfit.ToString("0.00"), m_MinTotalProfit.ToString("0.00"));
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

            UpdateInvestments(dailyAnalyzes, day);

            ReleaseInvestments(day);

            CreateNewInvestments(dailyAnalyzes, day);

            AnalyzeInvestments(day);

            DeleteReleasedInvestments();
        }

        private void AnalyzeInvestments(int day)
        {
            foreach (Investment investment in Investments)
            {
                m_InvestmentAnalyzis.Add(investment, day);
            }
        }

        private void DeleteReleasedInvestments()
        {
            List<Investment> investmentsToRelease = Investments.Where(x => x.IsEndOfInvestment).ToList();
            foreach (Investment investment in investmentsToRelease)
            {
                Investments.Remove(investment);
            }
        }

        private void UpdateInvestments(DailyAnalyzes dailyAnalyzes, int day)
        {
            foreach (Investment investment in Investments.OrderBy(x => x.ID))
            {
                investment.UpdateInvestment(dailyAnalyzes, day, TotalProfit, StocksTotalProfit[investment.DataSet.DataSetName]);
            }
        }

        private void ReleaseInvestments(int day)
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

        private void ReleaseInvestment(int day, Investment investment)
        {
            AccountBallance = investment.UpdateAccountOnRelease(day, AccountBallance);
            if (TotalProfit > m_MaxTotalProfit)
            {
                m_MaxTotalProfit = TotalProfit;
            }
            else if (TotalProfit < m_MinTotalProfit)
            {
                m_MinTotalProfit = TotalProfit;
            }

            StocksTotalProfit[investment.DataSet.DataSetName] = investment.Release(day, ref m_TotalProfit, StocksTotalProfit[investment.DataSet.DataSetName]);
            Log.AddMessage("Release investment of {0} with prediction {1}:", investment.DataSet.DataSetName, investment.PredictedChange.ToString());
            Log.AddMessage("AccountBalance {0}, release profit {1}, total profit {2}, correctness {3}, {4} predictions", AccountBallance.ToString("0.00"),
                investment.GetProfit(day).ToString("0.00"), TotalProfit.ToString("0.00"), investment.Analyze.AverageCorrectness.ToString("0.00"), investment.Analyze.NumOfPredictions);
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
            Investment investment = new Investment(DataSets[analyze.DataSetName], PriceDataSets[analyze.DataSetName], analyze, day, AccountBallance, TotalProfit, StocksTotalProfit[analyze.DataSet.DataSetName]);
            AccountBallance = investment.UpdateAccountOnInvestment(day, AccountBallance);
            if (TotalProfit > m_MaxTotalProfit)
            {
                m_MaxTotalProfit = TotalProfit;
            }
            else if (TotalProfit < m_MinTotalProfit)
            {
                m_MinTotalProfit = TotalProfit;
            }
            Log.AddMessage("New investment of {0} with prediction {1}, num of investments {2}:", investment.DataSet.DataSetName, investment.PredictedChange.ToString(), Investments.Count + 1);
            Log.AddMessage("Account balance {0}, {1} {2} shares, price {3}", AccountBallance, (investment.InvestmentType == BuySell.Buy) ? "bought" : "sold", investment.Ammount, investment.InvestedPrice);
            Investments.Add(investment);
            m_TotalNumOfInvestments++;
        }

        private void CreateNewInvestments(DailyAnalyzes dailyAnalyzes, int day)
        {
            if (day == 0)
            {
                return;
            }

            List<Analyze> analyzes = new List<Analyze>();
            foreach (DataSet dataSet in dailyAnalyzes.Keys)
            {
                analyzes.AddRange(dailyAnalyzes[dataSet].Values);
            }

            var orderAnalyzes = analyzes.OrderBy(x => x, new AnalyzeComparer());//OrderByDescending(x => x.AverageCorrectness);//

            foreach (Analyze analyze in orderAnalyzes)
            {
                if ((Investments.Count < MaxNumOfInvestments) && analyze.PredictedChange.Range <= MaxPredictedRange && analyze.PredictedChange.Range >= MinPredictedRange)
                {
                    AddInvestment(day, analyze);
                }
            }

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

        private List<PredictionRecord> GetRelevantPredictions(int day)
        {
            List<PredictionRecord> fitAnalyzerRecords = new List<PredictionRecord>();
            foreach (DataSet dataSet in m_PredictionRecords.Keys)
            {
                foreach (PredictionRecord predictionRecord in m_PredictionRecords[dataSet].Where(x => x.PredictionCorrectness > EffectivePredictionResult && x.Combination.Count >= MinCombinationItemsNum
                && x.Combination.Count <= MaxCombinationItemsNum))
                {
                    //bool[] goodPredictions = 
                    if (IsAnalyzeFits(day, predictionRecord))
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
                if (!DataPredictions[predictionRecord.DataSet.DataSetName].IsContainsPrediction(combinationItem, dataSetRow, PredictionErrorRange, -PredictionErrorRange))
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
