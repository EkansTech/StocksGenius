//using StocksData;
//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;

//namespace StocksSimulation
//{
//    public class StockSimulation
//    {
//        #region Members

//        private DateTime m_SimulationDate;

//        private double m_MaxTotalProfit = 0.0;

//        private double m_MinTotalProfit = 0.0;

//        private int m_TotalNumOfInvestments = 0;

//        InvestmentAnalyzis m_InvestmentAnalyzis;

//        private int m_SimulationRun = 0;

//        #endregion

//        #region Properties

//        public Dictionary<string, DataSet> DataSets { get; set; }

//        public Dictionary<string, DataSet> PriceDataSets { get; set; }

//        public double AccountBallance { get; set; }

//        private double m_TotalProfit;

//        public double TotalProfit
//        {
//            get { return m_TotalProfit; }
//            set { m_TotalProfit = value; }
//        }

//        public Dictionary<string, double> StocksTotalProfit { get; set; }


//        internal List<Investment> Investments { get; set; }

//        internal List<Investment> InvestmentsToRemove { get; set; }

//        static public string WorkingDirectory { get; set; }

//        static public int InvestmentsPerStock { get; set; }

//        public static double MinimumChange { get; set; }

//        #endregion

//        #region Constructors

//        public StockSimulation(DataSetsMetaData metaData, string workingDirectory)
//        {
//            WorkingDirectory = workingDirectory;
//            string dataSetsFolder = workingDirectory + DSSettings.DataSetsDir;
//            string priceDataSetsFolder = workingDirectory + DSSettings.PriceDataSetsDir;
//            DataSets = new Dictionary<string, DataSet>();
//            PriceDataSets = new Dictionary<string, DataSet>();
//            m_InvestmentAnalyzis = new InvestmentAnalyzis(workingDirectory, 0);
//            StocksTotalProfit = new Dictionary<string, double>();

//            foreach (string dataSetCode in metaData.Keys)
//            {
//                DataSet dataSet = new DataSet(dataSetCode, metaData[dataSetCode].DataSetFilePath, TestDataAction.LoadOnlyTestData);
//                DataSets.Add(dataSet.DataSetCode, dataSet);

//                DataSet priceDataSet = new DataSet(dataSetCode, metaData[dataSetCode].PriceDataSetFilePath, TestDataAction.LoadOnlyTestData);
//                PriceDataSets.Add(priceDataSet.DataSetCode, priceDataSet);
//                StocksTotalProfit.Add(dataSet.DataSetCode, 0.0);
//            }

//            Investments = new List<Investment>();
//            InvestmentsToRemove = new List<Investment>();
//            AccountBallance = 0.0;
//            TotalProfit = 0.0;
//        }

//        #endregion

//        #region Interface

//        public void Simulate()
//        {
//            m_MaxTotalProfit = 0.0;
//            m_MinTotalProfit = 0.0;
//            m_TotalNumOfInvestments = 0;
//            Log.AddMessage("Simulating, Investment money: {0}", AccountBallance);

//            m_MaxTotalProfit = 0.0;
//            m_MinTotalProfit = 0.0;
//            AccountBallance = 0.0;
//            TotalProfit = 0.0;

//            SimRecorder simRecorder = new SimRecorder(m_SimulationRun);

//            foreach (string dataSetName in StocksTotalProfit.Keys)
//            {
//                StocksTotalProfit[dataSetName] = 0.0;
//            }

//            for (int dataSetRow = DSSettings.TestRange; dataSetRow >= 0; dataSetRow--)
//            {
//                m_SimulationDate = new DateTime((long)DataSets.Values.First().GetDayData(dataSetRow)[0]);
//                Log.AddMessage("Trade date: {0}", m_SimulationDate.ToShortDateString());
//                RunSimulationCycle(dataSetRow);
//                simRecorder.AddRecord(dataSetRow, m_SimulationDate, AccountBallance, TotalProfit);
//            }
//            m_SimulationRun++;
//            m_InvestmentAnalyzis.SimulationRun++;
//            simRecorder.SaveToFile("Stock", WorkingDirectory + SimSettings.SimulationRecordsDirectory, m_MaxTotalProfit, m_MinTotalProfit, m_TotalNumOfInvestments);

//            m_InvestmentAnalyzis.SaveToFileNoPredictions();

//            Log.AddMessage("Final ammount of money: {0}", AccountBallance);
//            Log.AddMessage("Max total profit = {0}, min total profit = {1}", m_MaxTotalProfit.ToString("0.00"), m_MinTotalProfit.ToString("0.00"));
//        }

//        #endregion

//        #region Private Methods
        
//        private void RunSimulationCycle(int day)
//        {
//            Log.AddMessage("{0}:", m_SimulationDate.ToShortDateString());

//            UpdateInvestments(day);

//            ReleaseInvestments(day);

//            CreateNewInvestments(day);

//            AnalyzeInvestments(day);

//            DeleteReleasedInvestments();
//        }

//        private void AnalyzeInvestments(int day)
//        {
//            foreach (Investment investment in Investments)
//            {
//                m_InvestmentAnalyzis.Add(investment, day);
//            }
//            foreach (Investment investment in InvestmentsToRemove)
//            {
//                m_InvestmentAnalyzis.Add(investment, day);
//            }
//        }

//        private void DeleteReleasedInvestments()
//        {
//            InvestmentsToRemove.Clear();
//        }

//        private void UpdateInvestments(int dataSetRow)
//        {
//            foreach (Investment investment in Investments)
//            {
//                investment.UpdateInvestment(dataSetRow, TotalProfit, StocksTotalProfit[investment.DataSet.DataSetCode]);

//                if (investment.IsEndOfInvestment)
//                {
//                    InvestmentsToRemove.Add(investment);
//                }
//            }

//            foreach (Investment investment in InvestmentsToRemove)
//            {
//                Investments.Remove(investment);
//            }
//        }

//        private void ReleaseInvestments(int day)
//        {
//            foreach (Investment investment in InvestmentsToRemove)
//            {
//                ReleaseInvestment(day, investment);
//            }
//        }

//        private void ReleaseInvestment(int day, Investment investment)
//        {
//            AccountBallance = investment.UpdateAccountOnRelease(day, AccountBallance);
//            if (TotalProfit > m_MaxTotalProfit)
//            {
//                m_MaxTotalProfit = TotalProfit;
//            }
//            else if (TotalProfit < m_MinTotalProfit)
//            {
//                m_MinTotalProfit = TotalProfit;
//            }

//            StocksTotalProfit[investment.DataSet.DataSetCode] = investment.Release(day, ref m_TotalProfit, StocksTotalProfit[investment.DataSet.DataSetCode]);
//            Log.AddMessage("Release investment of {0}:", investment.DataSet.DataSetCode);
//            Log.AddMessage("AccountBalance {0}, release profit {1}, total profit {2}", AccountBallance.ToString("0.00"), investment.GetProfit(day).ToString("0.00"), TotalProfit.ToString("0.00"));
//        }

//        private void AddInvestment(int day, DataSet dataSet, BuySell buySell)
//        {
//            Investment investment = new Investment(dataSet, day, AccountBallance, TotalProfit, StocksTotalProfit[dataSet.DataSetCode], buySell);
//            AccountBallance = investment.UpdateAccountOnInvestment(day, AccountBallance);
//            if (TotalProfit > m_MaxTotalProfit)
//            {
//                m_MaxTotalProfit = TotalProfit;
//            }
//            else if (TotalProfit < m_MinTotalProfit)
//            {
//                m_MinTotalProfit = TotalProfit;
//            }
//            Log.AddMessage("New investment of {0}, num of investments {1}:", investment.DataSet.DataSetCode, Investments.Count + 1);
//            Log.AddMessage("Account balance {0}, {1} {2} shares, price {3}", AccountBallance, (investment.InvestmentType == BuySell.Buy) ? "bought" : "sold", investment.Ammount, investment.InvestedPrice);
//            Investments.Add(investment);
//            m_TotalNumOfInvestments++;
//        }

//        private void CreateNewInvestments(int day)
//        {
//            if (day == 0)
//            {
//                return;
//            }

//            foreach (DataSet dataSet in DataSets.Values)
//            {
//                BuySell investmentType = Investment.TimeToInvest(dataSet, Investments, day);
//                if (investmentType == BuySell.Buy)
//                {
//                    AddInvestment(day, dataSet, BuySell.Buy);
//                }
//                else if (investmentType == BuySell.Sell)
//                {
//                    AddInvestment(day, dataSet, BuySell.Sell);
//                }
//            }
//        }

//        #endregion
//    }

//}
