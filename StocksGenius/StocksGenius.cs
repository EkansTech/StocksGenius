using StocksData;
using StocksSimulation;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public class StocksGenius
    {
        #region Members

        private StocksData.StocksData m_StocksData = new StocksData.StocksData(SGSettings.Workspace, SGSettings.DataSourceType);
        private List<PredictionRecord> m_RelevantPredictionsRecords;

        #endregion

        #region Properties

        #endregion

        #region Constructor

        public StocksGenius()
        {
        }

        #endregion

        #region Interface

        public void UpdateDataSets()
        {
            m_StocksData.DataSource.UpdateDataSets(SGSettings.Workspace, m_StocksData.MetaData);
        }

        public void BuildPredictions()
        {
            m_StocksData.BuildDataPredictions();
        }

        public void BuildCombinedPredictions()
        {
            m_StocksData.BuildCombinedDataPredictions();
        }
        

        public void BuildSimPredictions(string suffix = null)
        {
            m_StocksData.BuildSimDataPredictions(SGSettings.PredictionsSince, SGSettings.PredictionEveryX, SGSettings.PredictionEveryXType,  SGSettings.PredictionsUpTo, SGSettings.RelevantDataType, SGSettings.DataRelevantX, SGSettings.DataRelevantXType, suffix);
        }

        public void GetActions()
        {
            m_RelevantPredictionsRecords = m_StocksData.LoadPredictions(SGSettings.EffectivePredictionResult);
            m_RelevantPredictionsRecords = GetRelevantPredictions();
            
            DailyAnalyzes dailyAnalyzes = GetPredictionsConclusions();
            dailyAnalyzes.RemoveBadAnalyzes();


            foreach (DataSetAnalyzes dataSetAnalyze in dailyAnalyzes.Values)
            {
                Console.WriteLine("DataSet {0}:", dataSetAnalyze.DataSetName);
                foreach (Analyze analyze in dataSetAnalyze.Values.OrderBy(x => x.PredictedChange.Range))
                {
                    Console.WriteLine("Analyzed prediction {0}, num of predictions {1}, average correctness {2}", analyze.PredictedChange.ToString(), analyze.NumOfPredictions, analyze.AverageCorrectness);
                }
            }
        }

        public void Simulate()
        {
            PredictionsSimulator predictionsSimulator = new PredictionsSimulator(m_StocksData.MetaData, SGSettings.Workspace);
            //analyzerSimulator.TestAnalyzeResults(stocksDataPath + iForexTestAnalyzerFolder);
            //Log.ConnectToConsole = false;
            predictionsSimulator.Simulate();

            //Console.Write(Log.ToString());
            //Log.SaveLogToFile(SGSettings.Workspace + "PredictionsSimulator.log");

            SimRecorder.SaveSummary(SGSettings.Workspace, "PredicrionSimSummary");

            return;
        }

        public void SimulateCombinedPredictions()
        {
            CombinedPredictionsSimulator combinedPredictionsSimulator = new CombinedPredictionsSimulator(m_StocksData.MetaData, SGSettings.Workspace);
            combinedPredictionsSimulator.Simulate();

            //Console.Write(Log.ToString());
            Log.SaveLogToFile(SGSettings.Workspace + "CombinedPredictionsSimulator.log");

            SimRecorder.SaveSummary(SGSettings.Workspace, "CombinedPredictionsSimSummary");
        }

        public void SimulateModel()
        {
            StockSimulation stockSimulation = new StockSimulation(m_StocksData.MetaData, SGSettings.Workspace);
            //analyzerSimulator.TestAnalyzeResults(stocksDataPath + iForexTestAnalyzerFolder);
            //Log.ConnectToConsole = false;
            stockSimulation.Simulate();

            //Console.Write(Log.ToString());
            //Log.SaveLogToFile(SGSettings.Workspace + "StocksSimulation.log");


            SimRecorder.SaveSummary(SGSettings.Workspace, "StockSimSummary");

            return;
        }

        public void AnalyzePredictions()
        {
            PredictionsAnalyze.AnalyzePredictions(SGSettings.Workspace, m_StocksData.MetaData);
        }

        public void RunInvestor(bool useSimPredictions = false)
        {
            if (m_StocksData.UseSimPredictions != useSimPredictions)
            {
                m_StocksData = new StocksData.StocksData(SGSettings.Workspace, SGSettings.DataSourceType, useSimPredictions);
            }
            Investor investor = new Investor(m_StocksData, useSimPredictions);
            investor.RunInvestor();
        }

        #endregion

        #region Private Methods

        private DailyAnalyzes GetPredictionsConclusions(int dataSetRow = 0)
        {
            DailyAnalyzes conclusions = new DailyAnalyzes();
            List<PredictionRecord> relevantPredictions = GetRelevantPredictions(dataSetRow);

            foreach (PredictionRecord record in relevantPredictions)
            {
                conclusions.Add(record.DataSet.DataSetCode, record.PredictedChange, record);
            }

            return conclusions;
        }

        private List<PredictionRecord> GetRelevantPredictions(int dataSetRow = 0)
        {
            List<PredictionRecord> fitAnalyzerRecords = new List<PredictionRecord>();

            foreach (PredictionRecord predictionRecord in m_RelevantPredictionsRecords)
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
                if (!m_StocksData.DataPredictions[predictionRecord.DataSet.DataSetCode].IsContainsPrediction(combinationItem, dataSetRow))
                {
                    return false;
                }
            }

            return true;
        }


        #endregion
    }
}
