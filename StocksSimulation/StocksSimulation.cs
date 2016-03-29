//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using StocksData;

//namespace StocksSimulation
//{
//    public class StocksSimulation
//    {
//        #region Members

//        private List<Dictionary<int, List<double>>> m_CalculatedPredictions = null;

//        private int m_HoldingSharesAmmount = 0;

//        #endregion
        
//        #region Properties

//        public DataSet DataSet { get; set; }

//        public AnalyzesDataSet AnalyzesDataSet { get; set; }

//        public double InvestedMoney { get; set; }

//        public List<Investment> Investments { get; set; }

//        #endregion

//        #region Constructors

//        public StocksSimulation(DataSet dataSet, AnalyzesDataSet analyzesDataSet)
//        {
//            DataSet = dataSet;
//            AnalyzesDataSet = analyzesDataSet;
//            Investments = new List<Investment>();
//        }

//        #endregion

//        #region Interface

//        public void Simulate()
//        {

//            for (int i = 2; i < 200; i++)
//            {
//                List<AnalyzerRecord> analyzeRecords = AnalyzesDataSet.GetBestPredictions(Constants.EffectivePredictionResult);
//                CalculatePredictions(analyzeRecords, i);

//                InvestedMoney = 10000;

//                Console.WriteLine("Simulating: {0}, Investment money: {1}", DataSet.DataSetName, InvestedMoney);
                        
//                for (int dataSetRow = i; dataSetRow > 0; dataSetRow--)
//                {
//                    RunSimulationCycle(dataSetRow, analyzeRecords);
//                }

//                Console.WriteLine("Final ammount of money: {0}", InvestedMoney);

//            }
//        }

//        #endregion

//        #region Private Methods

//        private void CalculatePredictions(List<AnalyzerRecord> analyzeRecords, int simulationRange)
//        {
//            m_CalculatedPredictions = new List<Dictionary<int, List<double>>>();
//            for (int rowNumber = 0; rowNumber <= simulationRange; rowNumber++)
//            {
//                Dictionary<int, List<double>> predictions = new Dictionary<int, List<double>>();
//                foreach (AnalyzerRecord analyzeRecord in analyzeRecords)
//                {
//                    if (!predictions.ContainsKey(analyzeRecord.Depth))
//                    {
//                        predictions.Add(analyzeRecord.Depth, new List<double>());
//                        CalculatePredictions(rowNumber, analyzeRecord.Depth, predictions[analyzeRecord.Depth]);
//                    }
//                }
//                m_CalculatedPredictions.Add(predictions);
//            }
//        }

//        private void CalculatePredictions(int rowNumber, int depth, List<double> predictions)
//        {
//            for (int i = 0; i < (int)PredictionsDataSet.DataColumns.NumOfColumns; i++)
//            {
//                predictions.Add(0.0);
//            }

//            for (int currentRow = rowNumber; currentRow < rowNumber + depth; currentRow++)
//            {
//                predictions[(int)PredictionsDataSet.DataColumns.OpenChange] += CalculateChange(currentRow + 1, currentRow + 2, (int)DataSet.DataColumns.Open, (int)DataSet.DataColumns.Open);
//                predictions[(int)PredictionsDataSet.DataColumns.HighChange] += CalculateChange(currentRow + 1, currentRow + 2, (int)DataSet.DataColumns.High, (int)DataSet.DataColumns.High);
//                predictions[(int)PredictionsDataSet.DataColumns.LowChange] += CalculateChange(currentRow + 1, currentRow + 2, (int)DataSet.DataColumns.Low, (int)DataSet.DataColumns.Low);
//                predictions[(int)PredictionsDataSet.DataColumns.CloseChange] += CalculateChange(currentRow + 1, currentRow + 2, (int)DataSet.DataColumns.Close, (int)DataSet.DataColumns.Close);
//                predictions[(int)PredictionsDataSet.DataColumns.VolumeChange] += CalculateChange(currentRow + 1, currentRow + 2, (int)DataSet.DataColumns.Volume, (int)DataSet.DataColumns.Volume);
//                predictions[(int)PredictionsDataSet.DataColumns.HighLowDif] += CalculateChange(currentRow + 1, currentRow + 1, (int)DataSet.DataColumns.High, (int)DataSet.DataColumns.Low);
//                predictions[(int)PredictionsDataSet.DataColumns.HighOpenDif] += CalculateChange(currentRow + 1, currentRow + 1, (int)DataSet.DataColumns.High, (int)DataSet.DataColumns.Open);
//                predictions[(int)PredictionsDataSet.DataColumns.LowOpenDif] += CalculateChange(currentRow + 1, currentRow + 1, (int)DataSet.DataColumns.Low, (int)DataSet.DataColumns.Open);
//                predictions[(int)PredictionsDataSet.DataColumns.CloseOpenDif] += CalculateChange(currentRow + 1, currentRow + 1, (int)DataSet.DataColumns.Close, (int)DataSet.DataColumns.Open);
//                predictions[(int)PredictionsDataSet.DataColumns.HighPrevCloseDif] += CalculateChange(currentRow + 1, currentRow + 2, (int)DataSet.DataColumns.High, (int)DataSet.DataColumns.Close);
//                predictions[(int)PredictionsDataSet.DataColumns.LowPrevCloseDif] += CalculateChange(currentRow + 1, currentRow + 2, (int)DataSet.DataColumns.Low, (int)DataSet.DataColumns.Close);
//                predictions[(int)PredictionsDataSet.DataColumns.OpenPrevCloseDif] += CalculateChange(currentRow + 1, currentRow + 2, (int)DataSet.DataColumns.Open, (int)DataSet.DataColumns.Close);
//            }
//        }

//        private double CalculateChange(int newDataRow, int originalDataRow, int newDataColumn, int originalDataColumn)
//        {
//            return (DataSet[newDataRow * (int)DataSet.DataColumns.NumOfColumns + newDataColumn] - DataSet[originalDataRow * (int)DataSet.DataColumns.NumOfColumns + originalDataColumn]) / DataSet[originalDataRow * (int)DataSet.DataColumns.NumOfColumns + originalDataColumn];
//        }

//        private void RunSimulationCycle(int dataSetRow, List<AnalyzerRecord> analyzeRecords)
//        {
//            List<double> investmentData = DataSet.GetDayData(dataSetRow);
//            List<double> nextDayData = DataSet.GetDayData(dataSetRow + 1);

//            AnalyzerRecord analyzeRecord = SelectBestPrediction(dataSetRow, analyzeRecords);
            
//            if (analyzeRecord == null)
//            {
//                return;
//            }

//            if (analyzeRecord.AnalyzedChange == AnalyzedChange.Up)
//            {
//                if (analyzeRecord.PredictedChange == ChangesDataSet.DataColumns.CloseOpenDif)
//                {
//                    int numOfShares = (int)(InvestedMoney / (investmentData[(int)DataSet.DataColumns.Open] * Constants.BuyActionPenalty));
//                    InvestedMoney -= (int)(numOfShares * (investmentData[(int)DataSet.DataColumns.Open] * Constants.BuyActionPenalty));
//                    InvestedMoney += (int)(numOfShares * (investmentData[(int)DataSet.DataColumns.Close] * Constants.SellActionPenalty));
//                }
//                else if (analyzeRecord.PredictedChange == ChangesDataSet.DataColumns.OpenPrevCloseDif)
//                {
//                    int numOfShares = (int)(InvestedMoney / (investmentData[(int)DataSet.DataColumns.Close] * Constants.BuyActionPenalty));
//                    InvestedMoney -= (int)(numOfShares * (investmentData[(int)DataSet.DataColumns.Close] * Constants.BuyActionPenalty));
//                    InvestedMoney += (int)(numOfShares * (nextDayData[(int)DataSet.DataColumns.Open] * Constants.SellActionPenalty));
//                }
//            }
//            else if(analyzeRecord.AnalyzedChange == AnalyzedChange.Down)
//            {
//                if (analyzeRecord.PredictedChange == ChangesDataSet.DataColumns.CloseOpenDif)
//                {
//                    int numOfShares = (int)(InvestedMoney / (investmentData[(int)DataSet.DataColumns.Open] * Constants.SellActionPenalty));
//                    InvestedMoney += (int)(numOfShares * (investmentData[(int)DataSet.DataColumns.Open] * Constants.SellActionPenalty));
//                    InvestedMoney -= (int)(numOfShares * (investmentData[(int)DataSet.DataColumns.Close] * Constants.BuyActionPenalty));
//                }
//                else if (analyzeRecord.PredictedChange == ChangesDataSet.DataColumns.OpenPrevCloseDif)
//                {
//                    int numOfShares = (int)(InvestedMoney / (investmentData[(int)DataSet.DataColumns.Close] * Constants.SellActionPenalty));
//                    InvestedMoney += (int)(numOfShares * (investmentData[(int)DataSet.DataColumns.Close] * Constants.SellActionPenalty));
//                    InvestedMoney -= (int)(numOfShares * (nextDayData[(int)DataSet.DataColumns.Open] * Constants.BuyActionPenalty));
//                }
//            }
//        }

//        private AnalyzerRecord SelectBestPrediction(int dataSetRow, List<AnalyzerRecord> analyzeRecords)
//        {
//            foreach (AnalyzerRecord analyzeRecord in analyzeRecords)
//            {
//                if (IsAnalyzeFits(dataSetRow, analyzeRecord))
//                {
//                    return analyzeRecord;
//                }
//            }

//            return null;
//        }

//        private bool IsAnalyzeFits(int dataSetRow, AnalyzerRecord analyzeRecord)
//        {
//            foreach (AnalyzesDataSet.AnalyzeCombination combinationItem in AnalyzesDataSet.CombinationItems)
//            {
//                if ((combinationItem & analyzeRecord.Combination) == combinationItem)
//                {
//                    PredictionsDataSet.DataColumns predictionColumn;
//                    bool positiveChange;
//                    GetPredictionType(out positiveChange, out predictionColumn, combinationItem);
//                    if ((positiveChange && m_CalculatedPredictions[dataSetRow][analyzeRecord.Depth][(int)predictionColumn] <= -StocksData.Constants.PredictionErrorRange)
//                        || (!positiveChange && m_CalculatedPredictions[dataSetRow][analyzeRecord.Depth][(int)predictionColumn] >= StocksData.Constants.PredictionErrorRange))
//                    {
//                        return false;
//                    }
//                }
//            }

//            return true;
//        }

//        private void GetPredictionType(out bool positiveChange, out PredictionsDataSet.DataColumns predictionColumn, AnalyzesDataSet.AnalyzeCombination combinationItem)
//        {
//            positiveChange = true;
//            predictionColumn = PredictionsDataSet.DataColumns.OpenChange;
//            switch (combinationItem)
//            {
//                case AnalyzesDataSet.AnalyzeCombination.OpenChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.OpenChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.HighChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.HighChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.LowChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.LowChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.CloseChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.CloseChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.VolumeChange: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.VolumeChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.HighLowDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.HighLowDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.HighOpenDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.HighOpenDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.LowOpenDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.LowOpenDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.CloseOpenDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.CloseOpenDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.HighPrevCloseDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.HighPrevCloseDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.LowPrevCloseDif:  { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.LowPrevCloseDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.OpenPrevCloseDif: { positiveChange = true; predictionColumn = PredictionsDataSet.DataColumns.OpenPrevCloseDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeOpenChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.OpenChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeHighChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.HighChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeLowChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.LowChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeCloseChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.CloseChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeVolumeChange: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.VolumeChange; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeHighLowDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.HighLowDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeHighOpenDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.HighOpenDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeLowOpenDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.LowOpenDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeCloseOpenDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.CloseOpenDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeHighPrevCloseDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.HighPrevCloseDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeLowPrevCloseDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.LowPrevCloseDif; } break;
//                case AnalyzesDataSet.AnalyzeCombination.NegativeOpenPrevCloseDif: { positiveChange = false; predictionColumn = PredictionsDataSet.DataColumns.OpenPrevCloseDif; } break;
//            }
//        }

//        #endregion
//    }
//}
