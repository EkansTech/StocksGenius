using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class AnalyzerSimulator
    {
        #region Constants
        
        public const double EffectivePredictionResult = 0.68;

        public const double BuyActionPenalty = 1.002;

        public const double SellActionPenalty = 0.998;

        public const double Profit = 1.004;

        #endregion

        #region Members

        private int m_HoldingSharesAmmount = 0;

        #endregion

        #region Properties

        public DataSet DataSet { get; set; }

        public DataAnalyzer DataAnalyzer { get; set; }

        public double AccountBallance { get; set; }

        public Investment Investment { get; set; }

        #endregion

        #region Constructors

        public AnalyzerSimulator(DataSet dataSet, DataAnalyzer dataAnalyzer)
        {
            DataSet = dataSet;
            DataAnalyzer = dataAnalyzer;
            Investment = null;
        }

        #endregion

        #region Interface

        public void Simulate()
        {
            List<AnalyzerRecord> analyzerRecords = DataAnalyzer.GetBestPredictions(EffectivePredictionResult);

            AccountBallance = 10000;

            Console.WriteLine("Simulating: {0}, Investment money: {1}", DataSet.DataSetName, AccountBallance);

            for (int dataSetRow = DataAnalyzer.TestRange; dataSetRow >= 0; dataSetRow--)
            {
                RunSimulationCycle(dataSetRow, analyzerRecords);
            }

            Console.WriteLine("Final ammount of money: {0}", AccountBallance);
        }

        #endregion

        #region Private Methods

        private void RunSimulationCycle(int dataSetRow, List<AnalyzerRecord> analyzerRecords)
        {
            List<double> todayData = DataSet.GetDayData(dataSetRow);

            List<AnalyzerRecord> relevantAnalyzerRecords = GetRelevantPredictions(dataSetRow, analyzerRecords);
            
            if (dataSetRow == 0)
            {
                if (Investment != null && (Investment.InvestedDay - Investment.AnalyzerRecord.PredictedChange.Range) < 0)
                {
                    AccountBallance += Investment.Ammount * todayData[(int)DataSet.DataColumns.Close];
                    Investment = null;
                }
            }


            bool onlyCloseOperations = false;

            if (Investment != null)
            {
                double change = (Investment.Price - todayData[(int)DataSet.DataColumns.Open]) / Investment.Price;
                if (Investment.InvestedDay - Investment.AnalyzerRecord.PredictedChange.Range == dataSetRow
                    || change > (BuyActionPenalty * BuyActionPenalty) && Investment.Ammount > 0
                    || change < (SellActionPenalty * SellActionPenalty) && Investment.Ammount < 0)
                {
                    if (Investment.AnalyzerRecord.PredictedChange.DataItem == DataItem.CloseOpenDif || Investment.AnalyzerRecord.PredictedChange.DataItem == DataItem.NegativeCloseOpenDif)
                    {
                        AccountBallance += Investment.Ammount * todayData[(int)DataSet.DataColumns.Close];
                        onlyCloseOperations = true;
                        Investment = null;
                    }
                    else
                    {
                        AccountBallance += Investment.Ammount * todayData[(int)DataSet.DataColumns.Open];
                        Investment = null;
                    }
                }
                else
                {
                    return;
                }
            }

            if (relevantAnalyzerRecords.Count == 0)
            {
                return;
            }

            AnalyzerRecord analyzerRecord = relevantAnalyzerRecords.OrderByDescending(x => x.PredictionCorrectness).FirstOrDefault();

            if (dataSetRow == 0)
            {

                if (analyzerRecord.PredictedChange.Range > 1 && analyzerRecord.PredictedChange.DataItem != DataItem.CloseOpenDif && analyzerRecord.PredictedChange.DataItem != DataItem.NegativeCloseOpenDif)
                {
                    return;
                }
            }

            if (analyzerRecord == null)
            {
                return;
            }

            if (analyzerRecord.PredictedChange.DataItem == DataItem.CloseOpenDif && !onlyCloseOperations)
            {
                int numOfShares = (int)(AccountBallance / (todayData[(int)DataSet.DataColumns.Open] * BuyActionPenalty));
                if (analyzerRecord.PredictedChange.Range == 1)
                {
                    Investment = new Investment() { Ammount = numOfShares, AnalyzerRecord = analyzerRecord, InvestedDay = dataSetRow, Price = todayData[(int)DataSet.DataColumns.Open] };
                    AccountBallance -= (int)(numOfShares * (todayData[(int)DataSet.DataColumns.Open] * BuyActionPenalty));
                }
                else
                {
                    AccountBallance -= (int)(numOfShares * (todayData[(int)DataSet.DataColumns.Open] * BuyActionPenalty));
                    AccountBallance += (int)(numOfShares * (todayData[(int)DataSet.DataColumns.Close] * SellActionPenalty));
                }
            }
            else if (analyzerRecord.PredictedChange.DataItem == DataItem.OpenPrevCloseDif)
            {
                int numOfShares = (int)(AccountBallance / (todayData[(int)DataSet.DataColumns.Close] * BuyActionPenalty));
                Investment = new Investment() { Ammount = numOfShares, AnalyzerRecord = analyzerRecord, InvestedDay = dataSetRow, Price = todayData[(int)DataSet.DataColumns.Close] };
                AccountBallance -= (int)(numOfShares * (todayData[(int)DataSet.DataColumns.Close] * BuyActionPenalty));
            }
            else if (analyzerRecord.PredictedChange.DataItem == DataItem.NegativeCloseOpenDif && !onlyCloseOperations)
            {
                int numOfShares = -(int)(AccountBallance / (todayData[(int)DataSet.DataColumns.Open] * SellActionPenalty));
                if (analyzerRecord.PredictedChange.Range == 1)
                {
                    Investment = new Investment() { Ammount = numOfShares, AnalyzerRecord = analyzerRecord, InvestedDay = dataSetRow, Price = todayData[(int)DataSet.DataColumns.Open] };
                    AccountBallance -= (int)(numOfShares * (todayData[(int)DataSet.DataColumns.Open] * BuyActionPenalty));
                }
                else
                {
                    AccountBallance -= (int)(numOfShares * (todayData[(int)DataSet.DataColumns.Open] * BuyActionPenalty));
                    AccountBallance += (int)(numOfShares * (todayData[(int)DataSet.DataColumns.Close] * SellActionPenalty));
                }
            }
            else if (analyzerRecord.PredictedChange.DataItem == DataItem.NegativeOpenPrevCloseDif)
            {
                int numOfShares = -(int)(AccountBallance / (todayData[(int)DataSet.DataColumns.Close] * SellActionPenalty));
                Investment = new Investment() { Ammount = numOfShares, AnalyzerRecord = analyzerRecord, InvestedDay = dataSetRow, Price = todayData[(int)DataSet.DataColumns.Close] };
                AccountBallance -= (int)(numOfShares * (todayData[(int)DataSet.DataColumns.Close] * SellActionPenalty));
            }
        }

        private List<AnalyzerRecord> GetRelevantPredictions(int dataSetRow, List<AnalyzerRecord> analyzerRecords)
        {
            List<AnalyzerRecord> fitAnalyzerRecords = new List<AnalyzerRecord>();
            foreach (AnalyzerRecord analyzerRecord in analyzerRecords)
            {
                if (IsAnalyzeFits(dataSetRow, analyzerRecord))
                {
                    fitAnalyzerRecords.Add(analyzerRecord);
                }
            }

            return fitAnalyzerRecords;
        }

        private bool IsAnalyzeFits(int dataSetRow, AnalyzerRecord analyzerRecord)
        {
            foreach (CombinationItem combinationItem in analyzerRecord.Combination)
            {
                if (!DataAnalyzer.IsContainsPrediction(combinationItem, dataSetRow, -DataAnalyzer.PredictionErrorRange, DataAnalyzer.PredictionErrorRange))
                {
                    return false;
                }
            }

            return true;
        }

        private void GetPredictionType(out bool positiveChange, out PredictionsDataSet.DataColumns predictionColumn, AnalyzesDataSet.AnalyzeCombination combinationItem)
        {
            positiveChange = true;
            predictionColumn = PredictionsDataSet.DataColumns.OpenChange;
            switch (combinationItem)
            {
                case AnalyzesDataSet.AnalyzeCombination.OpenChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.OpenChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.HighChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.HighChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.LowChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.LowChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.CloseChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.CloseChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.VolumeChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.VolumeChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.HighLowDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.HighLowDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.HighOpenDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.HighOpenDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.LowOpenDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.LowOpenDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.CloseOpenDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.CloseOpenDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.HighPrevCloseDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.HighPrevCloseDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.LowPrevCloseDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.LowPrevCloseDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.OpenPrevCloseDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.OpenPrevCloseDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeOpenChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.OpenChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeHighChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.HighChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeLowChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.LowChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeCloseChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.CloseChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeVolumeChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.VolumeChange; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeHighLowDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.HighLowDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeHighOpenDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.HighOpenDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeLowOpenDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.LowOpenDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeCloseOpenDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.CloseOpenDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeHighPrevCloseDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.HighPrevCloseDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeLowPrevCloseDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.LowPrevCloseDif; } break;
                case AnalyzesDataSet.AnalyzeCombination.NegativeOpenPrevCloseDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.OpenPrevCloseDif; } break;
            }
        }

        #endregion
    }
}
