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


        Dictionary<int, List<Investment>> SimulationData { get; set; }

        public int SimulationRun { get; set; }

        public string WorkingDirectory { get; set; }

        #endregion

        #region Constructors

        public InvestmentAnalyzis(string workingDirectory)
        {
            WorkingDirectory = workingDirectory;
            SimulationRun = 0;
            SimulationData = new Dictionary<int, List<Investment>>();
        }

        #endregion

        #region Interface

        public void Add(Investment investment)
        {
            if (!SimulationData.ContainsKey(SimulationRun))
            {
                SimulationData.Add(SimulationRun, new List<Investment>());
            }

            SimulationData[SimulationRun].Add(investment);
        }

        public void SaveToFile()
        {
            string filePath = string.Format("{0}\\{1}_{2}.csv", WorkingDirectory, FileName, DateTime.Now.ToString().Replace(':', '_').Replace('/', '_'));

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("SimulationRun,ReleseReason,InvestmentDay,ReleaseDay,InvestmentType,DataSetName,InvestedPrive,Profit,PredictedChange,NumOfPredictions,AverageCorrectness");
                foreach (int simulationRun in SimulationData.Keys)
                {
                    foreach (Investment investment in SimulationData[simulationRun])
                    {
                        writer.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}",
                            simulationRun,
                            investment.ReleaseReason,
                            investment.InvestmentDay,
                            investment.ReleaseDay,
                            investment.InvestmentType,
                            investment.DataSet.DataSetName,
                            investment.InvestedPrice,
                            investment.Profit,
                            investment.PredictedChange.ToString(),
                            investment.Analyze.NumOfPredictions,
                            investment.Analyze.AverageCorrectness);
                    }
                }
            }
        }

        #endregion

    }
}
