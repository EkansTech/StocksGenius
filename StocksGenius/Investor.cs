using StocksData;
using StocksSimulation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
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

        private DateTime m_Today = DateTime.Today; 

        public DateTime Today
        {
            get { return m_Today; }
            set { m_Today = value; }
        }

        public double InvestmentsValue
        {
            get
            {
                return ActiveInvestments.Sum(x => x.GetInvestmentValue(Today));
            }
        }

        public double TotalValue
        {
            get
            {
                return m_RealMoney + InvestmentsValue;
            }
        }

        public double MoneyToInvest
        {
            get
            {
                return m_RealMoney + ActiveInvestments.Sum(x => x.GetProfit(m_Today));// - ActiveInvestments.Sum(x => x.InvestedMoney * SGSettings.SafesForStockRate);
            }
        }


        #endregion

        #region Constructors

        public Investor(StocksData.StocksData stocksData)
        {
            StocksData = stocksData;
            stocksData.LoadDataSets();

            foreach (string dataSetName in stocksData.DataSets.Keys)
            {
                m_StocksTotalProfit.Add(dataSetName, 0.0);
            }

            LoadInvestorState();
            GetTodayOpenData();
            StocksData.AddOpenDataToDataSets(SGSettings.DataSetsCodesPrefix);
        }

        #endregion

        #region Interface

        public void RunInvestor()
        {
            if (!StocksData.AreDataSetsSynchronized())
            {
                Console.WriteLine("Error: Not all datasets are synchronized");
            }

            bool exit = false;

            while (!exit)
            {
                Console.WriteLine("Select an action:");
                Console.WriteLine("1. Show Active Investments");
                Console.WriteLine("2. Release Investments");
                Console.WriteLine("3. New Investment Actions");
                Console.WriteLine("4. Save Investor State");
                Console.WriteLine("5. Go To Date");
                Console.WriteLine("6. Show Status");
                Console.WriteLine("7. Get Today Open Data");
                Console.WriteLine("8. ResetInvestorState");
                Console.WriteLine("0. Back");

                string input = Console.ReadLine();

                switch (input)
                {
                    case "1": ShowActiveInvestments(); break;
                    case "2": ReleaseInvestments(); break;
                    case "3": NewInvestments(); break;
                    case "4": SaveInvestorState(); break;
                    case "5": GoToDate(); break;
                    case "6": ShowStatus(); break;
                    case "7": GetTodayOpenData(); break;
                    case "8": ResetInvestorState(); break;
                    case "0": exit = true; break;
                    default:
                        break;
                }
            }

            SaveInvestorState();
        }

        private void ResetInvestorState()
        {
            m_LastTradeDate = DateTime.MinValue;
            m_RealMoney = 2500.0;

            ActiveInvestments = new Investments();
            ReleasedInvestments = new Investments();
            Investment.Reset(0, 0);

            foreach (string dataSetName in StocksData.DataSets.Keys)
            {
                m_StocksTotalProfit[dataSetName] = 0.0;
            }
        }

        private void GetTodayOpenData()
        {
            //Dictionary<string, double> openData = 
            //string dataSetCodes = string.Empty;
            //foreach (string dataSetCode in StocksData.MetaData.Keys)
            //{
            //    dataSetCodes += dataSetCode.Replace("WIKI-", string.Empty) + "+";
            //}

            //dataSetCodes = dataSetCodes.TrimEnd('+');

            //using (var client = new WebClient())
            //{
            //    client.DownloadFile(string.Format("http://finance.yahoo.com/d/quotes.csv?s={0}&f=sod1", dataSetCodes), DSSettings.Workspace + SGSettings.NewOpenDataFile);
            //}
        }

        private void ShowStatus()
        {
            Console.WriteLine("Num of investments: {0}, Real Money: {1}, Investments value: {2}, Total value {3},  MoneyToInvest {4}", NumOfInvestments, m_RealMoney, InvestmentsValue, TotalValue, MoneyToInvest);
            Console.WriteLine("Stocks Profit:");
            foreach (string dataSetName in m_StocksTotalProfit.Keys)
            {
                Console.WriteLine("{0} - {1}", dataSetName, m_StocksTotalProfit[dataSetName]);
            }
        }

        private void LoadInvestorState()
        {
            IniFile investorSettings = new IniFile(DSSettings.Workspace + SGSettings.InvestorIni);

            m_LastTradeDate = investorSettings.IniReadDateTime("General", "LastTradeDate");
            m_RealMoney = investorSettings.IniReadDoubleValue("General", "RealMoney");

            Investments investments = new Investments();
            investments.LoadFromFile(SGSettings.InvestmentsFileName);

            ActiveInvestments = new Investments(investments, InvestmentStatus.Active);
            ReleasedInvestments = new Investments(investments, InvestmentStatus.Released);

            foreach (string dataSetName in StocksData.DataSets.Keys)
            {
                double value = m_StocksTotalProfit[dataSetName];
                investorSettings.IniReadDoubleValue("StocksProfit", dataSetName, ref value);
                m_StocksTotalProfit[dataSetName] = value;
            }

            m_Today = m_LastTradeDate;
        }

        private void SaveInvestorState()
        {
            m_LastTradeDate = m_Today;
            IniFile investorSettings = new IniFile(DSSettings.Workspace + SGSettings.InvestorIni);

            investorSettings.IniWriteValue("General", "TotalValue", TotalValue);
            investorSettings.IniWriteValue("General", "LastTradeDate", m_LastTradeDate);
            investorSettings.IniWriteValue("General", "RealMoney", m_RealMoney);

            Investments investments = new Investments(ActiveInvestments, InvestmentStatus.Active);
            if (ReleasedInvestments != null)
            {
                investments.AddRange(ReleasedInvestments);
            }
            investments.SaveToFile(m_Today, SGSettings.InvestmentsFile);

            foreach (string dataSetName in m_StocksTotalProfit.Keys)
            {
                investorSettings.IniWriteValue("StocksProfit", dataSetName, m_StocksTotalProfit[dataSetName]);
            }
        }

        public void ShowActiveInvestments()
        {
            foreach (Investment investment in ActiveInvestments)
            {
                Console.WriteLine("ID: {0}, DS: {1}, Type: {2}, Ammount: {3}, Current Value: {4}, Num Of {5}: {6}, Live length: {7}", investment.ID, StocksData.MetaData[investment.DataSetCode].Name, investment.InvestmentType, 
                    investment.Ammount, investment.GetInvestmentValue(m_Today), investment.InvestmentType == BuySell.Buy ? "Downs" : "Ups", investment.Analyze.NumOfPredictions, investment.GetLiveLength(m_Today));
            }
        }

        private void ReleaseInvestments()
        {
            List<string> dataSets = GetTradableDataSets();
            foreach (Investment activeInvestment in ActiveInvestments)
            {
                double profitRatio = activeInvestment.GetProfit(m_Today) / activeInvestment.InvestedMoney;
                bool isTradable = dataSets.FirstOrDefault(x => x == activeInvestment.DataSetCode) != null;
                Console.WriteLine("{9}ID: {0}, DS: {1}, Type: {2}, Ammount: {3}, Current Profit: {4}, Num Of {5}: {6}, Live length: {7}, Profit {8}%", activeInvestment.ID, StocksData.MetaData[activeInvestment.DataSetCode].Name,
                activeInvestment.InvestmentType, activeInvestment.Ammount, activeInvestment.GetProfit(m_Today), activeInvestment.InvestmentType == BuySell.Buy ? "Downs" : "Ups", 
                activeInvestment.Analyze.NumOfPredictions, activeInvestment.GetLiveLength(m_Today), (profitRatio * 100.0).ToString("0.00"), isTradable ? string.Empty : "NotTradable ");
            }

            Console.WriteLine("Num of investments: {0}, Real Money: {1}, Investments value: {2}, Total value {3},  MoneyToInvest {4}", NumOfInvestments, m_RealMoney, InvestmentsValue, TotalValue, MoneyToInvest);

            Console.WriteLine("Enter investment ID to Release");
            int id = Convert.ToInt32(Console.ReadLine());
            Investment investment = ActiveInvestments.FirstOrDefault(x => x.ID == id);
            if (investment == null)
            {
                Console.WriteLine("Wrong Investment ID");
                return;
            }

            m_StocksTotalProfit[investment.DataSetCode] = investment.Release(m_Today, m_StocksTotalProfit[investment.DataSetCode]);
            investment.UpdateRealMoneyOnRelease(m_Today, ref m_RealMoney);

            ActiveInvestments.Remove(investment);
            ReleasedInvestments.Add(investment);

            Log.AddMessage("Released investment of {0}: {1}, Num of {2} {3}, Change of {4}%, Profit {5}", StocksData.MetaData[investment.DataSetCode].Name, investment.InvestmentType, 
                investment.Analyze.IsPositiveInvestment ? "Downs" : "Ups", investment.Analyze.NumOfPredictions, (investment.Analyze.AverageCorrectness * 100).ToString("0.00"), investment.GetProfit(m_Today));
            Console.WriteLine("Num of investments: {0}, Real Money: {1}, Investments value: {2}, Total value {3},  MoneyToInvest {4}", NumOfInvestments, m_RealMoney, InvestmentsValue, TotalValue, MoneyToInvest);
        }

        private void NewInvestments()
        {
            List<string> dataSets = GetTradableDataSets();

            List<Analyze> potentialInvestments = new List<Analyze>();
            foreach (string dataSetCode in dataSets)
            {
                DataSet dataSet = StocksData.DataSets[dataSetCode];
                for (int numOfDowns = 0; numOfDowns < 100; numOfDowns++)
                {
                    int dayNum = dataSet.GetDayNum(m_Today);
                    if (dataSet.NumOfRows > dayNum + numOfDowns + 1 && (dataSet.GetData(dayNum + numOfDowns, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfDowns + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfDowns + 1, DataSet.DataColumns.Open) < 0)
                    {
                    }
                    else
                    {
                        if (numOfDowns >= SGSettings.MinNumOfDowns || (dataSet.GetData(dayNum, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfDowns, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfDowns, DataSet.DataColumns.Open) < SGSettings.MinChangeForDown)
                        {
                            Analyze analyze = new Analyze()
                            {
                                AverageCorrectness = (dataSet.GetData(dayNum, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfDowns, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfDowns, DataSet.DataColumns.Open),
                                DataSetCode = dataSet.DataSetCode,
                                NumOfPredictions = numOfDowns,
                                PredictedChange = new CombinationItem(1, DataItem.OpenUp, 0, 0)
                            };
                            potentialInvestments.Add(analyze);
                        }
                        else
                        {

                        }
                        break;
                    }
                }

                for (int numOfUps = 0; numOfUps < 100; numOfUps++)
                {
                    int dayNum = dataSet.GetDayNum(m_Today);
                    if (dataSet.NumOfRows > dayNum + numOfUps + 1 && (dataSet.GetData(dayNum + numOfUps, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfUps + 1, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfUps + 1, DataSet.DataColumns.Open) > 0)
                    {
                    }
                    else
                    {
                        if (numOfUps >= SGSettings.MinNumOfUps || (dataSet.GetData(dayNum, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfUps, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfUps, DataSet.DataColumns.Open) > SGSettings.MinChangeForUp)
                        {
                            Analyze analyze = new Analyze()
                            {
                                DataSetCode = dataSet.DataSetCode,
                                NumOfPredictions = numOfUps,
                                PredictedChange = new CombinationItem(1, DataItem.OpenDown, 0, 0),
                                AverageCorrectness = (dataSet.GetData(dayNum, DataSet.DataColumns.Open) - dataSet.GetData(dayNum + numOfUps, DataSet.DataColumns.Open)) / dataSet.GetData(dayNum + numOfUps, DataSet.DataColumns.Open),
                            };
                            potentialInvestments.Add(analyze);
                        }
                        else
                        {

                        }
                        break;
                    }
                }        
            }

            potentialInvestments = potentialInvestments.OrderByDescending(x => (x.AverageCorrectness > 0 ? x.AverageCorrectness : -x.AverageCorrectness + 10) * 100 + x.NumOfPredictions).ToList();
            Dictionary<int, Analyze> actionsMap = potentialInvestments.ToDictionary(x => potentialInvestments.IndexOf(x) + 1);
            foreach (int analyzeNum in actionsMap.Keys)
            {
                Analyze analyze = actionsMap[analyzeNum];
                Console.WriteLine("{0}: DS {1}, {2}, Num of {3} {4}, Change of {5}%", analyzeNum, StocksData.MetaData[analyze.DataSetCode].Name, analyze.IsPositiveInvestment ? "Buy" : "Sell",
                    analyze.IsPositiveInvestment ? "Downs" : "Ups", analyze.NumOfPredictions, (analyze.AverageCorrectness * 100).ToString("0.00"));
            }

            Console.WriteLine("Num of investments: {0}, Real Money: {1}, Investments value: {2}, Total value {3},  MoneyToInvest {4}", NumOfInvestments, m_RealMoney, InvestmentsValue, TotalValue, MoneyToInvest);

            Console.WriteLine("Select Investment Action:");
            int action = Convert.ToInt32(Console.ReadLine());

            if (!actionsMap.ContainsKey(action))
            {
                Console.WriteLine("Error: Wrong Action Number");
                return;
            }

            double y = SGSettings.InvestmentPerStock;
            Console.WriteLine("Select Money: 1. {0}, 2. {1}, 3. {2}, 4. {3}, 5. {4}, 6. {5}, 7. {6}, 8. {7}, 9. {8}, 10. {9}",
                y, y * 2, y * 4, y * 8, y * 16, y * 32, y * 64, y * 128, y * 256, y * 512);

            double investingMoney = Convert.ToInt32(Console.ReadLine());
            for (int i = 0; i < investingMoney - 1; i++)
            {
                y *= 2;
            }
            Analyze selectedAnalyze = actionsMap[action];

            Investment investment = new Investment(selectedAnalyze, m_Today, m_StocksTotalProfit[selectedAnalyze.DataSetCode], y);

            investment.UpdateRealMoneyOnInvestment(m_Today, ref m_RealMoney);
            ActiveInvestments.Add(investment);

            Console.WriteLine("New investment of {0}: {1}, Num of {2} {3}, Change of {4}%, Profit {5}", StocksData.MetaData[investment.DataSetCode].Name, investment.InvestmentType,
                investment.Analyze.IsPositiveInvestment ? "Downs" : "Ups", investment.Analyze.NumOfPredictions, (investment.Analyze.AverageCorrectness * 100).ToString("0.00"), investment.GetProfit(m_Today));
            Console.WriteLine("Num of investments: {0}, Real Money: {1}, Investments value: {2}, Total value {3},  MoneyToInvest {4}", NumOfInvestments, m_RealMoney, InvestmentsValue, TotalValue, MoneyToInvest);
        }
        private void GoToDate()
        {
            DateTime dayRef = m_Today.AddDays(1);
            int i = 0;
            while (i < 20 && StocksData.DataSets.Values.Where(x => x.ContainsTradeDay(dayRef)).Count() == 0)
            {
                dayRef = dayRef.AddDays(1);
                i++;
            }

            i = 0;
            Dictionary<int, DateTime> daysMap = new Dictionary<int, DateTime>();
            while (i < 20)
            {
                while (StocksData.DataSets.Values.Where(x => x.ContainsTradeDay(dayRef)).Count() == 0)
                {
                    dayRef = dayRef.AddDays(-1);
                }

                daysMap.Add(i, dayRef);
                Console.WriteLine("{0}. {1}", i, dayRef.ToShortDateString());
                i++;
                dayRef = dayRef.AddDays(-1);
            }

            Console.WriteLine("Last trade date: {0}", m_LastTradeDate.ToShortDateString());

            Console.WriteLine("Select date: ");
            int dayNum = Convert.ToInt32(Console.ReadLine());
            m_LastTradeDate = m_Today;
            m_Today = daysMap[dayNum];

            Console.WriteLine("Current date is {0}", m_Today.ToShortDateString());
        }

        #endregion

        #region Private Methods

        private List<string> GetTradableDataSets()
        {
            List<string> tradableDataSets = new List<string>();
            foreach (string dataSetCode in StocksData.DataSets.Keys)
            {
                if (StocksData.DataSets[dataSetCode].IsTradableDay(m_Today))
                {
                    tradableDataSets.Add(dataSetCode);
                }
            }

            return tradableDataSets;
        }

        #endregion
    }
}
