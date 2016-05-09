using StocksData;
using StocksSimulation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public class Investor
    {
        #region Properties

        public Investments ActiveInvestments { get; set; }

        public Investments ReleasedInvestments { get; set; }

        public static StocksData.StocksData StocksData { get; set; }

        private double m_AccountBallance;

        public double AccountBalance
        {
            get { return m_AccountBallance; }
            set { m_AccountBallance = value; }
        }

        private double m_TotalProfit;

        public double TotalProfit
        {
            get { return m_TotalProfit; }
            set { m_TotalProfit = value; }
        }

        private double m_RealMoney;

        public double RealMoney
        {
            get { return m_RealMoney; }
            set { m_RealMoney = value; }
        }


        private Dictionary<string, double> m_StocksTotalProfit = new Dictionary<string, double>();

        public Dictionary<string, double> StocksTotalProfit
        {
            get { return m_StocksTotalProfit; }
            set { m_StocksTotalProfit = value; }
        }

        public int NumOfInvestments
        {
            get { return ActiveInvestments.Count; }
        }

        private DateTime m_LastTradeDate;

        public DateTime LastTradeDate
        {
            get { return m_LastTradeDate; }
            set { m_LastTradeDate = value; }
        }

        private DailyAnalyzes m_DailyAnalyzes;

        public DailyAnalyzes DailyAnalyzes
        {
            get { if (m_DailyAnalyzes == null) { m_DailyAnalyzes = GetPredictionsConclusions(); } return m_DailyAnalyzes; }
            set { m_DailyAnalyzes = value; }
        }

        #endregion

        #region Constructors

        public Investor(StocksData.StocksData stocksData)
        {
            StocksData = stocksData;
            StocksData.ReloadDataSets();

            foreach (string dataSetName in stocksData.DataSets.Keys)
            {
                m_StocksTotalProfit.Add(dataSetName, 0.0);
            }

            LoadInvestorState();
        }

        #endregion

        #region Interface

        public void RunInvestor()
        {
            Console.Clear();

            if (!StocksData.AreDataSetsSynchronized())
            {
                Console.WriteLine("Error: Not all datasets are synchronized");
            }

            bool exit = false;

            while (!exit)
            {
                Console.WriteLine("Select an action:");
                Console.WriteLine("1. Show Active Investments");
                Console.WriteLine("2. Show Predictions");
                Console.WriteLine("3. Release Investments");
                Console.WriteLine("4. New Investment Actions");
                Console.WriteLine("5. Save Investor State");
                Console.WriteLine("6. Go To Date");
                Console.WriteLine("7. Show Status");
                Console.WriteLine("0. Back");

                string input = Console.ReadLine();

                switch (input)
                {
                    case "1": ShowActiveInvestments(); break;
                    case "2": ShowPredictions(); break;
                    case "3": ReleaseInvestments(); break;
                    case "4": NewInvestments(); break;
                    case "5": SaveInvestorState(); break;
                    case "6": GoToDate(); break;
                    case "7": ShowStatus(); break;
                    case "0": exit = true; break;
                    default:
                        break;
                }
            }

            SaveInvestorState();
        }

        private void ShowStatus()
        {
            double investmentsValue = ActiveInvestments.Sum(x => x.GetInvestmentValue()); 
            Console.WriteLine("Account Ballance: {0}, Num of investments: {1}, Total profit: {2}", m_AccountBallance, NumOfInvestments, m_TotalProfit);
            Console.WriteLine("Real Money: {0}, Investments value: {1}, Total value {2}", m_RealMoney, investmentsValue, m_RealMoney + investmentsValue);
            Console.WriteLine("Stocks Profit:");
            foreach (string dataSetName in m_StocksTotalProfit.Keys)
            {
                Console.WriteLine("{0} - {1}", dataSetName, m_StocksTotalProfit[dataSetName]);
            }
        }

        private void LoadInvestorState()
        {
            IniFile investorSettings = new IniFile(SGSettings.WorkingDirectory + SGSettings.InvestorIni);

            m_AccountBallance = investorSettings.IniReadDoubleValue("General", "AccountBalance");
            m_TotalProfit = investorSettings.IniReadDoubleValue("General", "TotalProfit");
            m_LastTradeDate = investorSettings.IniReadDateTime("General", "LastTradeDate");
            m_RealMoney = investorSettings.IniReadDoubleValue("General", "RealMoney");

            Investments investments = new Investments();
            investments.LoadFromFile(SGSettings.InvestmentsFileName);

            ActiveInvestments = new Investments(investments, InvestmentStatus.Active);
            ReleasedInvestments = new Investments(investments, InvestmentStatus.Released);

            foreach (string dataSetName in StocksData.DataSets.Keys)
            {
                m_StocksTotalProfit[dataSetName] = investorSettings.IniReadDoubleValue("StocksProfit", dataSetName);
            }

        }

        private void SaveInvestorState()
        {
            m_LastTradeDate = StocksData.DataSets.Values.First().GetDate(0);
            IniFile investorSettings = new IniFile(SGSettings.WorkingDirectory + SGSettings.InvestorIni);

            investorSettings.IniWriteValue("General", "AccountBalance", m_AccountBallance);
            investorSettings.IniWriteValue("General", "TotalProfit", m_TotalProfit);
            investorSettings.IniWriteValue("General", "LastTradeDate", m_LastTradeDate);
            investorSettings.IniWriteValue("General", "RealMoney", m_RealMoney);

            Investments investments = new Investments(ActiveInvestments, InvestmentStatus.Active);
            investments.AddRange(ReleasedInvestments);
            investments.SaveToFile(SGSettings.InvestmentsFile);

            foreach (string dataSetName in m_StocksTotalProfit.Keys)
            {
                investorSettings.IniWriteValue("StocksProfit", dataSetName, m_StocksTotalProfit[dataSetName]);
            }
        }

        public void ShowActiveInvestments()
        {
            foreach (Investment investment in ActiveInvestments)
            {
                Console.WriteLine("ID: {0}, DS: {1}, Type: {2}, Ammount: {3}, Current Profit: {4}, PredictedChange: {5}, Live length: {6}", 
                    investment.ID, investment.DataSetName, investment.InvestmentType, investment.Ammount, investment.GetCurrentProfit(), investment.PredictedChange.ToString(), investment.GetLiveLength());
            }
        }

        public void ShowPredictions()
        {
            Console.WriteLine("Predictions:");
            foreach (string dataSetName in DailyAnalyzes.Keys)
            {
                Console.WriteLine(dataSetName + ":");
                foreach (CombinationItem predictedChange in DailyAnalyzes[dataSetName].Keys.OrderBy(x => x.Range))
                {
                    Console.WriteLine("\tRange: {0}, Change: {1}, Average Corectness: {2}, Num Of Predictions: {3}", predictedChange.Range, predictedChange.DataItem,
                        DailyAnalyzes[dataSetName][predictedChange].AverageCorrectness, DailyAnalyzes[dataSetName][predictedChange].NumOfPredictions);
                }
            }
        }

        private void ReleaseInvestments()
        {
            foreach (Investment activeInvestment in ActiveInvestments)
            {
                Console.WriteLine("ID: {0}, DS: {1}, Type: {2}, Ammount: {3}, Current Profit: {4}, PredictedChange: {5}, Live length: {6}",
                    activeInvestment.ID, activeInvestment.DataSetName, activeInvestment.InvestmentType, activeInvestment.Ammount, activeInvestment.GetCurrentProfit(),
                    activeInvestment.PredictedChange.ToString(), activeInvestment.GetLiveLength());
                
                if (!DailyAnalyzes.ContainsKey(activeInvestment.DataSetName))
                {
                    Console.WriteLine("No Predictions");
                    continue;
                }

                foreach (Analyze analyze in DailyAnalyzes[activeInvestment.DataSetName].Values.OrderBy(x => x.PredictedChange.Range))
                {
                    Console.WriteLine("\tRange: {0}, Change: {1}, Average Corectness: {2}, Num Of Predictions: {3}", analyze.PredictedChange.Range, analyze.PredictedChange.DataItem,
                        analyze.AverageCorrectness, analyze.NumOfPredictions);
                }
            }

            Console.WriteLine("Enter investment ID to Release");
            int id = Convert.ToInt32(Console.ReadLine());
            Investment investment = ActiveInvestments.FirstOrDefault(x => x.ID == id);
            if (investment == null)
            {
                Console.WriteLine("Wrong Investment ID");
                return;
            }

            m_AccountBallance = investment.UpdateAccountOnRelease(m_AccountBallance);
            m_StocksTotalProfit[investment.DataSetName] = investment.Release(ref m_TotalProfit, m_StocksTotalProfit[investment.DataSetName]);
            investment.UpdateRealMoneyOnRelease(ref m_RealMoney);

            Log.AddMessage("Released investment of {0} with prediction {1}:", investment.DataSetName, investment.PredictedChange.ToString());
            Log.AddMessage("AccountBalance {0}, release profit {1}, total profit {2}, correctness {3}, {4} predictions", m_AccountBallance.ToString("0.00"),
                investment.GetProfit().ToString("0.00"), TotalProfit.ToString("0.00"), investment.Analyze.AverageCorrectness.ToString("0.00"), investment.Analyze.NumOfPredictions);

            ActiveInvestments.Remove(investment);
            ReleasedInvestments.Add(investment);
        }

        private void NewInvestments()
        {
            List<Analyze> analyzes = new List<Analyze>();
            foreach (string dataSetName in DailyAnalyzes.Keys)
            {
                analyzes.AddRange(DailyAnalyzes[dataSetName].Values);
            }

            List<Analyze> orderAnalyzes = analyzes.OrderBy(x => x.PredictedChange.Range).ToList();//OrderByDescending(x => x.AverageCorrectness);//

            Console.WriteLine("Proposed Actions:");
            for (int i = 0; i < orderAnalyzes.Count; i++)
            {
                Analyze analyze = orderAnalyzes[i];
                Console.WriteLine("{0}. {1} {2}, Prediction: {3},Average Corectness: {4}, Num Of Predictions: {5}, Investments for stock: {6}", i + 1, (analyze.IsPositiveInvestment) ? "Buy" : "Sell",
                    analyze.DataSetName, analyze.PredictedChange.ToString(), analyze.AverageCorrectness, analyze.NumOfPredictions, ActiveInvestments.Where(x => x.DataSetName.Equals(analyze.DataSetName)).Count());
            }

            Console.WriteLine("Select Action:");
            int action = Convert.ToInt32(Console.ReadLine()) - 1;
            Console.WriteLine();
            
            if (action < 0 || action >= orderAnalyzes.Count)
            {
                Console.WriteLine("Error: Wrong Action Number");
                return;
            }

            Analyze selectedAnalyze = orderAnalyzes[action];
            DataSet priceDataSet = StocksData.PriceDataSets[selectedAnalyze.DataSetName];

            Investment investment = new Investment(selectedAnalyze, priceDataSet.GetDate(0), m_AccountBallance, m_TotalProfit, m_StocksTotalProfit[priceDataSet.DataSetName]);

            m_AccountBallance = investment.UpdateAccountOnInvestment(m_AccountBallance);
            investment.UpdateRealMoneyOnInvestment(ref m_RealMoney);
            ActiveInvestments.Add(investment);

            Console.WriteLine("New investment of {0} with prediction {1}, num of investments {2}:", investment.DataSetName, investment.PredictedChange.ToString(), ActiveInvestments.Count);
            Console.WriteLine("Account balance {0}, {1} {2} shares, price {3}", m_AccountBallance, (investment.InvestmentType == BuySell.Buy) ? "bought" : "sold", investment.Ammount, investment.InvestedPrice);
        }
        private void GoToDate()
        {
            for (int i = 0; i < 20; i++)
            {
                Console.WriteLine("{0}. {1}", i, new DateTime((long)StocksData.DataSets.Values.First().GetData(i, DataSet.DataColumns.Date)).ToShortDateString());
            }

            Console.WriteLine("Select date: ");
            int dayNum = Convert.ToInt32(Console.ReadLine());

            StocksData.MoveToDate(dayNum);

            if (!StocksData.AreDataSetsSynchronized())
            {
                Console.WriteLine("Error: Not all datasets are synchronized");
            }
            else
            {
                Console.WriteLine("Current date is {0}", StocksData.DataSets.Values.First().GetDate(0).ToShortDateString());
            }
        }

        #endregion

        #region Private Methods

        private DailyAnalyzes GetPredictionsConclusions()
        {
            DailyAnalyzes conclusions = new DailyAnalyzes();
            List<PredictionRecord> relevantPredictions = GetRelevantPredictions();

            foreach (PredictionRecord record in relevantPredictions)
            {
                conclusions.Add(record.DataSet.DataSetName, record.PredictedChange, record);
            }

            return conclusions;
        }

        private List<PredictionRecord> GetRelevantPredictions()
        {
            List<PredictionRecord> fitAnalyzerRecords = new List<PredictionRecord>();
            foreach (DataPredictions dataPredictions in StocksData.DataPredictions.Values)
            {
                foreach (PredictionRecord predictionRecord in dataPredictions.GetBestPredictions(SGSettings.EffectivePredictionResult))
                {
                    if (IsAnalyzeFits(predictionRecord))
                    {
                        fitAnalyzerRecords.Add(predictionRecord);
                    }
                }
            }

            return fitAnalyzerRecords;
        }

        private bool IsAnalyzeFits(PredictionRecord predictionRecord)
        {
            foreach (CombinationItem combinationItem in predictionRecord.Combination)
            {
                if (!predictionRecord.DataPredictions.IsContainsPrediction(combinationItem, 0, SGSettings.PredictionErrorRange, -SGSettings.PredictionErrorRange))
                {
                    return false;
                }
            }

            return true;
        }

        #endregion
    }
}
