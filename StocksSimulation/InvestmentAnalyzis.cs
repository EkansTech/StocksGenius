﻿using StocksData;
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

        Dictionary<DateTime, List<Investment>> SimulationData { get; set; }

        public int SimulationRun { get; set; }

        public string WorkingDirectory { get; set; }

        #endregion

        #region Constructors

        public InvestmentAnalyzis(string workingDirectory, int simulationRun)
        {
            WorkingDirectory = workingDirectory;
            SimulationRun = simulationRun;
            SimulationData = new Dictionary<DateTime, List<Investment>>();
        }

        #endregion

        #region Interface

        public void Add(Investment investment, DateTime day)
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
                SubDirectory = WorkingDirectory + m_DirectoryName + SimSettings.SimStartTime.ToString().Replace(':', '_').Replace('/', '_') + "\\";
                Directory.CreateDirectory(SubDirectory);
            }

            string filePath = string.Format("{0}\\{1}_{2}.csv", SubDirectory, FileName, SimulationRun);

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("SimulationDay,ID,Action,ActionReason,InvestmentDay,ReleaseDay,InvestmentType,DataSetName,InvestedPrice,TodayPrice,Profit,"
                    + "Profit%,TotalValue,TotalProfit,RealMoney,StockTotalProfit,DataItem,Range,SequenceLength,Change,InvestmentValue,LastChange");
                foreach (DateTime day in SimulationData.Keys)
                {
                    foreach (Investment investment in SimulationData[day])
                    {
                        writer.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}%,{12},{13},{14},{15},{16},{17},{18},{19},{20},{21}%",
                            day.ToShortDateString(),
                            investment.ID,
                            investment.Action,
                            investment.ActionReason,
                            investment.InvestmentDay,
                            investment.ReleaseDay,
                            investment.InvestmentType,
                            investment.DataSet.DataSetCode,
                            investment.InvestedPrice,
                            investment.GetDayPrice(day),
                            investment.Profit,
                            investment.ProfitPercentage(day),
                            investment.TotalValue,
                            investment.TotalProfit,
                            investment.RealMoney,
                            investment.StockTotalProfit,
                            investment.PredictedChange.DataItem,
                            investment.PredictedChange.Range,
                            investment.Analyze.SequenceLength,
                            investment.Analyze.Change,
                            investment.GetInvestmentValue(day),
                            investment.Analyze.LastChange * 100);
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
                     + "DataSetName,InvestedPrice,TodayPrice,Profit,Profit%,TotalValue,StockTotalProfit,DataItem,Range,NumOfPredictions,AverageCorrectness");
                foreach (DateTime day in SimulationData.Keys)
                {
                    foreach (Investment investment in SimulationData[day].OrderBy(x => x.DataSet.DataSetCode))
                    {
                        writer.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}%,{12},{13}",
                            day.ToShortDateString(),
                            investment.ID,
                            investment.Action,
                            investment.ActionReason,
                            investment.InvestmentDay,
                            investment.ReleaseDay,
                            investment.InvestmentType,
                            investment.DataSet.DataSetCode,
                            investment.InvestedPrice,
                            investment.GetDayPrice(day),
                            investment.Profit,
                            investment.ProfitPercentage(day),
                            investment.TotalValue,
                            investment.StockTotalProfit,
                            investment.PredictedChange.DataItem,
                            investment.PredictedChange.Range,
                            investment.Analyze.SequenceLength,
                            investment.Analyze.Change);
                    }
                }
            }
        }

        #endregion

    }
}
