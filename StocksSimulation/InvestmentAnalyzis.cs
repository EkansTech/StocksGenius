using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class InvestmentAnalyzis
    {
        #region Properties

        private string m_FileName = "SimulationAnalyzis";

        public string FileName
        {
            get { return m_FileName; }
            set { m_FileName = value; }
        }


        Dictionary<int, Dictionary<int, List<Investment>>> SimulationData { get; set; }

        public int SimulationRun { get; set; }

        public string WorkingDirectory { get; set; }

        #endregion

        #region Constructors

        public InvestmentAnalyzis(string workingDirectory)
        {
            WorkingDirectory = workingDirectory;
            SimulationRun = 0;
            SimulationData = new Dictionary<int, Dictionary<int, List<Investment>>>();
        }

        #endregion

        #region Interface

        public void Add(Investment investment, int day)
        {
            if (!SimulationData.ContainsKey(SimulationRun))
            {
                SimulationData.Add(SimulationRun, new Dictionary<int, List<Investment>>());
            }
            if (!SimulationData[SimulationRun].ContainsKey(day))
            {
                SimulationData[SimulationRun].Add(day, new  List<Investment>());
            }

            SimulationData[SimulationRun][day].Add(investment.Clone());
        }

        public void SaveToFile()
        {
            string filePath = string.Format("{0}\\{1}_{2}.csv", WorkingDirectory, FileName, DateTime.Now.ToString().Replace(':', '_').Replace('/', '_'));

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("SimulationRun,SimulationDay,InvestmentID,Action,ActionReason,InvestmentDay,ReleaseDay,InvestmentType," 
                     + "DataSetName,InvestedPrice,TodayPrice,Profit,CurrentProfit%,TotalProfit,DataItem,Range,NumOfPredictions,AverageCorrectness");
                foreach (int simulationRun in SimulationData.Keys)
                {
                    foreach (int day in SimulationData[simulationRun].Keys)
                    {
                        foreach (Investment investment in SimulationData[simulationRun][day].OrderBy(x => x.DataSet.DataSetName))
                        {
                            writer.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}%,{13},{14},{15},{16},{17}",
                                simulationRun,
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
                                investment.PredictedChange.DataItem,
                                investment.PredictedChange.Range,
                                investment.Analyze.NumOfPredictions,
                                investment.Analyze.AverageCorrectness);
                        }
                    }
                }
            }
        }

        public void SaveToFileNoPredictions()
        {
            string filePath = string.Format("{0}\\{1}_{2}.csv", WorkingDirectory, FileName, DateTime.Now.ToString().Replace(':', '_').Replace('/', '_'));

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("SimulationRun,SimulationDay,InvestmentID,Action,ActionReason,InvestmentDay,ReleaseDay,InvestmentType,"
                     + "DataSetName,InvestedPrice,TodayPrice,Profit,CurrentProfit%,TotalProfit");
                foreach (int simulationRun in SimulationData.Keys)
                {
                    foreach (int day in SimulationData[simulationRun].Keys)
                    {
                        foreach (Investment investment in SimulationData[simulationRun][day].OrderBy(x => x.DataSet.DataSetName))
                        {
                            writer.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}%,{13}",
                                simulationRun,
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
                                investment.ReleaseTotalProfit);
                        }
                    }
                }
            }
        }

        #endregion

    }
}
