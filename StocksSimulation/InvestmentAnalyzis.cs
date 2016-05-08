using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    internal class InvestmentAnalyzis
    {
        #region Properties

        private string m_FileName = "InvestmentAnalyzis";

        public string FileName
        {
            get { return m_FileName; }
            set { m_FileName = value; }
        }

        private string m_DirectoryName = "\\InvestmentAnaylis\\";

        public string DirectoryName
        {
            get { return m_DirectoryName; }
            set { m_DirectoryName = value; }
        }

        public static string SubDirectory { get; set; }

        Dictionary<int, List<Investment>> SimulationData { get; set; }

        public int SimulationRun { get; set; }

        public string WorkingDirectory { get; set; }

        #endregion

        #region Constructors

        public InvestmentAnalyzis(string workingDirectory, int simulationRun)
        {
            WorkingDirectory = workingDirectory;
            SimulationRun = simulationRun;
            SimulationData = new Dictionary<int, List<Investment>>();
        }

        #endregion

        #region Interface

        public void Add(Investment investment, int day)
        {
            if (!SimulationData.ContainsKey(day))
            {
                SimulationData.Add(day, new  List<Investment>());
            }

            SimulationData[day].Add(investment.Clone());
        }

        public void SaveToFile()
        {
            if (!Directory.Exists(WorkingDirectory + m_DirectoryName))
            {
                Directory.CreateDirectory(WorkingDirectory + m_DirectoryName);
            }

            if (SimulationRun == 0)
            {
                SubDirectory = WorkingDirectory + m_DirectoryName + DateTime.Now.ToString().Replace(':', '_').Replace('/', '_') + "\\";
                Directory.CreateDirectory(SubDirectory);
            }

            string filePath = string.Format("{0}\\{1}_{2}.csv", SubDirectory, FileName, SimulationRun);

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("SimulationDay,InvestmentID,Action,ActionReason,InvestmentDay,ReleaseDay,InvestmentType," 
                     + "DataSetName,InvestedPrice,TodayPrice,Profit,CurrentProfit%,TotalProfit,StockTotalProfit,DataItem,Range,NumOfPredictions,AverageCorrectness");
                foreach (int day in SimulationData.Keys)
                {
                    foreach (Investment investment in SimulationData[day])
                    {
                        writer.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}%,{12},{13},{14},{15},{16},{17}",
                            day,
                            investment.ID,
                            investment.Action,
                            investment.ActionReason,
                            investment.InvestmentDay,
                            investment.ReleaseDay,
                            investment.InvestmentType,
                            investment.DataSet.DataSetName,
                            investment.InvestedPrice,
                            investment.GetDayPrice(day),
                            investment.Profit,
                            investment.CurrentProfitPercentage(day),
                            investment.ReleaseTotalProfit,
                            investment.ReleaseStockTotalProfit,
                            investment.PredictedChange.DataItem,
                            investment.PredictedChange.Range,
                            investment.Analyze.NumOfPredictions,
                            investment.Analyze.AverageCorrectness);
                    }
                }
            }
        }

        public void SaveToFileNoPredictions()
        {
            if (!Directory.Exists(WorkingDirectory + m_DirectoryName))
            {
                Directory.CreateDirectory(WorkingDirectory + m_DirectoryName);
            }

            if (SimulationRun == 0)
            {
                SubDirectory = WorkingDirectory + m_DirectoryName + DateTime.Now.ToString().Replace(':', '_').Replace('/', '_') + "\\";
                Directory.CreateDirectory(SubDirectory);
            }

            string filePath = string.Format("{0}\\{1}_{2}.csv", SubDirectory, FileName, SimulationRun);

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("SimulationDay,InvestmentID,Action,ActionReason,InvestmentDay,ReleaseDay,InvestmentType,"
                     + "DataSetName,InvestedPrice,TodayPrice,Profit,CurrentProfit%,TotalProfit,StockTotalProfit,DataItem,Range,NumOfPredictions,AverageCorrectness");
                foreach (int day in SimulationData.Keys)
                {
                    foreach (Investment investment in SimulationData[day].OrderBy(x => x.DataSet.DataSetName))
                    {
                        writer.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}%,{12},{13}",
                            day,
                            investment.ID,
                            investment.Action,
                            investment.ActionReason,
                            investment.InvestmentDay,
                            investment.ReleaseDay,
                            investment.InvestmentType,
                            investment.DataSet.DataSetName,
                            investment.InvestedPrice,
                            investment.GetDayPrice(day),
                            investment.Profit,
                            investment.CurrentProfitPercentage(day),
                            investment.ReleaseTotalProfit,
                            investment.ReleaseStockTotalProfit);
                    }
                }
            }
        }

        #endregion

    }
}
