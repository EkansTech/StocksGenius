﻿using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public class StocksGenius
    {
        #region Members

        private StocksData.StocksData m_StocksData = new StocksData.StocksData(SGSettings.WorkingDirectory);
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
            m_StocksData.DataSource.UpdateDataSets(SGSettings.WorkingDirectory);
        }

        public void BuildPredictions()
        {
            m_StocksData.BuildDataPredictions();
        }

        public void GetActions()
        {
            m_RelevantPredictionsRecords = m_StocksData.LoadPredictions(SGSettings.EffectivePredictionResult);
            m_RelevantPredictionsRecords = GetRelevantPredictions();
            
            DailyAnalyzes dailyAnalyzes = GetPredictionsConclusions();
            dailyAnalyzes.RemoveBadAnalyzes();


            foreach (DataSetAnalyzes dataSetAnalyze in dailyAnalyzes.Values)
            {
                Console.WriteLine("DataSet {0}:", dataSetAnalyze.DataSet.DataSetName);
                foreach (Analyze analyze in dataSetAnalyze.Values.OrderByDescending(x => x.AverageCorrectness))
                {
                    Console.WriteLine("Analyzed prediction {0}, num of predictions {1}, average correctness {2}", analyze.PredictedChange.ToString(), analyze.NumOfPredictions, analyze.AverageCorrectness);
                }
            }
        }

        #endregion

        #region Private Methods

        private DailyAnalyzes GetPredictionsConclusions(int dataSetRow = 0)
        {
            DailyAnalyzes conclusions = new DailyAnalyzes();
            List<PredictionRecord> relevantPredictions = GetRelevantPredictions(dataSetRow);

            foreach (PredictionRecord record in relevantPredictions)
            {
                conclusions.Add(record.DataSet, record.PredictedChange, record);
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
                if (!m_StocksData.DataPredictions[predictionRecord.DataSet.DataSetName].IsContainsPrediction(combinationItem, dataSetRow, -DSSettings.PredictionErrorRange, DSSettings.PredictionErrorRange))
                {
                    return false;
                }
            }

            return true;
        }


        #endregion
    }
}
