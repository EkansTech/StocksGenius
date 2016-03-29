//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;

//namespace StocksData
//{
//    public class AnalyzesDataSet : List<double>
//    {
//        #region Enums

//        [Flags] public enum AnalyzeCombination
//        {
//            None = 0,
//            OpenChange = 1,
//            HighChange = OpenChange * 2,
//            LowChange = HighChange * 2,
//            CloseChange = LowChange * 2,
//            VolumeChange = CloseChange * 2,
//            HighLowDif = VolumeChange * 2,
//            HighOpenDif = HighLowDif * 2,
//            LowOpenDif = HighOpenDif * 2,
//            CloseOpenDif = LowOpenDif * 2,
//            HighPrevCloseDif = CloseOpenDif * 2,
//            LowPrevCloseDif = HighPrevCloseDif * 2,
//            OpenPrevCloseDif = LowPrevCloseDif * 2,
//            NegativeOpenChange = OpenPrevCloseDif * 2,
//            NegativeHighChange = NegativeOpenChange * 2,
//            NegativeLowChange = NegativeHighChange * 2,
//            NegativeCloseChange = NegativeLowChange * 2 * 2,
//            NegativeVolumeChange = NegativeCloseChange * 2,
//            NegativeHighLowDif = NegativeVolumeChange * 2,
//            NegativeHighOpenDif = NegativeHighLowDif * 2,
//            NegativeLowOpenDif = NegativeHighOpenDif * 2,
//            NegativeCloseOpenDif = NegativeLowOpenDif * 2,
//            NegativeHighPrevCloseDif = NegativeCloseOpenDif * 2,
//            NegativeLowPrevCloseDif = NegativeHighPrevCloseDif * 2,
//            NegativeOpenPrevCloseDif = NegativeLowPrevCloseDif * 2,
//        }

//        #endregion

//        #region Members

//        private static readonly List<AnalyzeCombination> m_CombinationItems = GetCombinationItems();

//        private static List<AnalyzeCombination> m_AnalyzeCombinations = InitializeAnalyzeCombinations();

//        #endregion

//        #region Properties

//        public int NumOfDataColumns = Constants.DepthsNum * Constants.AnalyzeChangesList.Count * 2;

//        public int NumOfDataRows
//        {
//            get { return Count / NumOfDataColumns; }
//        }

//        private string m_AnalyzeDataSetName = string.Empty;

//        public string AnalyzeDataSetName
//        {
//            get { return m_AnalyzeDataSetName; }
//            set { m_AnalyzeDataSetName = value; }
//        }

//        public ChangesDataSet ChangesDataSet { get; private set; }

//        public PredictionsDataSet PredictionsDataSet { get; private set; }

//        #endregion

//        #region Constructors

//        public AnalyzesDataSet()
//        {
//        }

//        public AnalyzesDataSet(string filePath)
//        {
//            LoadFromFile(filePath);
//            m_AnalyzeDataSetName = Path.GetFileNameWithoutExtension(filePath);
//        }

//        public AnalyzesDataSet(ChangesDataSet changesDataSet, PredictionsDataSet predictionsDataSet, bool useGPU = true)
//        {
//            ChangesDataSet = changesDataSet;
//            PredictionsDataSet = predictionsDataSet;

//            LoadFromPredictionsDataSet(useGPU);
//        }

//        #endregion

//        #region Interface

//        public void LoadFromPredictionsDataSet(bool useGPU)
//        {
//            m_AnalyzeDataSetName = PredictionsDataSet.PredictionDataSetName.Substring(0, PredictionsDataSet.PredictionDataSetName.IndexOf(Constants.PredictionDataSetSuffix)) + Constants.AnalyzeDataSetSuffix;

//            if (useGPU)
//            {
//                AnalyzeGPU();
//            }
//            else
//            {
//                AnalyzeCPU();
//            }
//        }

//        public void LoadFromPredictionsDataSetGPU(PredictionsDataSet predictionsDataSet)
//        {
//            m_AnalyzeDataSetName = predictionsDataSet.PredictionDataSetName.Substring(0, predictionsDataSet.PredictionDataSetName.IndexOf(Constants.PredictionDataSetSuffix)) + Constants.AnalyzeDataSetSuffix;
//        }

//        public void LoadFromFile(string filePath)
//        {
//            StreamReader csvFile = new StreamReader(filePath);

//            // Read the first line and validate correctness of columns in the data file
//            csvFile.ReadLine();

//            while (!csvFile.EndOfStream)
//            {
//                Add(csvFile.ReadLine());
//            }
//        }
//        public void SaveDataToFile(string folderPath)
//        {
//            using (StreamWriter csvFile = new StreamWriter(folderPath + "\\" + AnalyzeDataSetName + ".csv"))
//            {
//                // Write the first line
//                csvFile.WriteLine(GetColumnNamesString());
//                for (int currentDate = 0; currentDate < NumOfDataRows; currentDate++)
//                {
//                    csvFile.WriteLine(GetDataString(currentDate));
//                }
//            }
//        }

//        public void Add(string dataLine)
//        {
//            string[] data = dataLine.Split(',');

//            for (int column = 1; column < data.Length; column++)
//            {
//                Add(Convert.ToDouble(data[column]));
//            }
//        }

//        #endregion

//        #region Private Methods


//        public void AnalyzeCPU()
//        {
//            for (int combination = 0; combination < m_AnalyzeCombinations.Count; combination++)
//            {
//                DoAnalyzeCombination(m_AnalyzeCombinations[combination]);
//            }
//        }

//        public void AnalyzeGPU()
//        {
//            AddRange(GPUAnalyze.AnalyzePredictions(ChangesDataSet.ToArray(), PredictionsDataSet.ToArray(), m_AnalyzeCombinations.ToArray()));
//        }

//        private static List<AnalyzeCombination> GetCombinationItems()
//        {
//            List<AnalyzeCombination> combinationItems = typeof(AnalyzeCombination).GetEnumValues().Cast<AnalyzeCombination>().ToList();

//            combinationItems.Remove(combinationItems.First());

//            return combinationItems;
//        }

//        private static List<AnalyzeCombination> InitializeAnalyzeCombinations()
//        {
//            List<AnalyzeCombination> analyzeCombinations = new List<AnalyzeCombination>();
//            for (int i = 1; i <= Constants.AnalyzeMaxCombinationSize; i++)
//            {
//                InitializeAnalyzeCombinations(analyzeCombinations, i);
//            }

//            return analyzeCombinations;
//        }

//        private static void InitializeAnalyzeCombinations(List<AnalyzeCombination> analyzeCombinations, int combinationSize,
//            AnalyzeCombination currentCombination = AnalyzeCombination.None, int combinationPart = 0, int startPosition = 0)
//        {
//            if (combinationPart >= combinationSize)
//            {
//                analyzeCombinations.Add(currentCombination);
//                return;
//            }

//            for (int i = startPosition; i < m_CombinationItems.Count - (combinationSize - combinationPart - 1); i++)
//            {
//                currentCombination |= m_CombinationItems[i];

//                InitializeAnalyzeCombinations(analyzeCombinations, combinationSize, currentCombination, combinationPart + 1, i + 1);

//                currentCombination &= ~ m_CombinationItems[i];
//            }

//        }

//        private void DoAnalyzeCombination(AnalyzeCombination analyzeCombination)
//        {
//            int predictionRow = 0;
//            while (predictionRow < PredictionsDataSet.NumOfRows)
//            {
//                 AnalyzePredictionDepth(ref predictionRow, analyzeCombination);
//            }
//        }

//        private void AnalyzePredictionDepth(ref int predictionRow, AnalyzeCombination analyzeCombination)
//        {
//            double[] upCorrectPredictions = new double[Constants.AnalyzeChangesList.Count];
//            double[] downCorrectPredictions = new double[Constants.AnalyzeChangesList.Count];
//            double[] predictions = new double[Constants.AnalyzeChangesList.Count];

//            double[] upPredictionsResults = new double[Constants.AnalyzeChangesList.Count];
//            double[] downPredictionsResults = new double[Constants.AnalyzeChangesList.Count];

//            double[] relevantHistoriesNum = new double[Constants.AnalyzeChangesList.Count];

//            double currentDepth = PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns];
//            int depthStartRow = predictionRow;
//            int changesRow = 0;

//            for (int i = 0; i < Constants.AnalyzeChangesList.Count; i++)
//            {
//                upCorrectPredictions[i] = 0.0;
//                downCorrectPredictions[i] = 0.0;
//                predictions[i] = 0.0;
//            }

//            for (; predictionRow < PredictionsDataSet.NumOfRows && predictionRow < depthStartRow + Constants.RelevantHistory; predictionRow++)
//            {
//                for (int predictedChangeNum = 0; predictedChangeNum < Constants.AnalyzeChangesList.Count; predictedChangeNum++)
//                {
//                    double prediction = GetPredictionResult(predictionRow, analyzeCombination);
//                    double actualChange = ChangesDataSet[changesRow * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

//                    if (prediction > 0.0)
//                    {
//                        predictions[predictedChangeNum]++;

//                        if (actualChange > Constants.PredictionErrorRange)
//                        {
//                            upCorrectPredictions[predictedChangeNum]++;
//                        }
//                        if (actualChange < -Constants.PredictionErrorRange)
//                        {
//                            downCorrectPredictions[predictedChangeNum]++;
//                        }
//                    }
//                }

//                changesRow++;
//            }

//            for (int i = 0; i < Constants.AnalyzeChangesList.Count; i++)
//            {
//                if (predictions[i] > 0)
//                {
//                    upPredictionsResults[i] = upCorrectPredictions[i] / predictions[i];
//                    downPredictionsResults[i] = downCorrectPredictions[i] / predictions[i];
//                    relevantHistoriesNum[i] = 2;
//                }
//                else
//                {
//                    relevantHistoriesNum[i] = 0;
//                }
//            }

//            for (; predictionRow < PredictionsDataSet.NumOfRows && PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns] == currentDepth; predictionRow++)
//            {
//                for (int predictedChangeNum = 0; predictedChangeNum < Constants.AnalyzeChangesList.Count; predictedChangeNum++)
//                {
//                    double prediction = GetPredictionResult(predictionRow - Constants.RelevantHistory, analyzeCombination);
//                    double actualChange = ChangesDataSet[(changesRow - Constants.RelevantHistory) * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

//                    if (prediction > 0.0)
//                    {
//                        predictions[predictedChangeNum]--;

//                        if (actualChange > Constants.PredictionErrorRange)
//                        {
//                            upCorrectPredictions[predictedChangeNum]--;
//                        }
//                        if (actualChange < -Constants.PredictionErrorRange)
//                        {
//                            downCorrectPredictions[predictedChangeNum]--;
//                        }
//                    }

//                    prediction = GetPredictionResult(predictionRow, analyzeCombination);
//                    actualChange = ChangesDataSet[changesRow * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

//                    if (prediction > 0.0)
//                    {
//                        predictions[predictedChangeNum]++;

//                        if (actualChange > Constants.PredictionErrorRange)
//                        {
//                            upCorrectPredictions[predictedChangeNum]++;
//                        }
//                        if (actualChange < -Constants.PredictionErrorRange)
//                        {
//                            downCorrectPredictions[predictedChangeNum]++;
//                        }
//                    }

//                    if (predictions[predictedChangeNum] > 0)
//                    {
//                        if (relevantHistoriesNum[predictedChangeNum] > 0)
//                        {
//                            upPredictionsResults[predictedChangeNum] += (upCorrectPredictions[predictedChangeNum] / predictions[predictedChangeNum] - upPredictionsResults[predictedChangeNum]) / relevantHistoriesNum[predictedChangeNum];
//                            downPredictionsResults[predictedChangeNum] += (downCorrectPredictions[predictedChangeNum] / predictions[predictedChangeNum] - downPredictionsResults[predictedChangeNum]) / relevantHistoriesNum[predictedChangeNum];
//                            relevantHistoriesNum[predictedChangeNum]++;
//                        }
//                        else
//                        {
//                            upPredictionsResults[predictedChangeNum] = upCorrectPredictions[predictedChangeNum] / predictions[predictedChangeNum];
//                            downPredictionsResults[predictedChangeNum] = downCorrectPredictions[predictedChangeNum] / predictions[predictedChangeNum];
//                            relevantHistoriesNum[predictedChangeNum] = 2;
//                        }
//                    }
//                }

//                changesRow++;
//            }

//            AddRange(upPredictionsResults);
//            AddRange(downPredictionsResults);
//        }

//        private double GetPredictionResult(int predictionsRow, AnalyzeCombination analyzeCombination)
//        {
//            double combinedPrediction = 1.0;
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.OpenChange) == AnalyzesDataSet.AnalyzeCombination.OpenChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenChange] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighChange) == AnalyzesDataSet.AnalyzeCombination.HighChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighChange] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.LowChange) == AnalyzesDataSet.AnalyzeCombination.LowChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowChange] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.CloseChange) == AnalyzesDataSet.AnalyzeCombination.CloseChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseChange] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.VolumeChange) == AnalyzesDataSet.AnalyzeCombination.VolumeChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.VolumeChange] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighLowDif) == AnalyzesDataSet.AnalyzeCombination.HighLowDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighLowDif] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighOpenDif) == AnalyzesDataSet.AnalyzeCombination.HighOpenDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighOpenDif] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.LowOpenDif) == AnalyzesDataSet.AnalyzeCombination.LowOpenDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowOpenDif] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.CloseOpenDif) == AnalyzesDataSet.AnalyzeCombination.CloseOpenDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseOpenDif] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.HighPrevCloseDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighPrevCloseDif] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.LowPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.LowPrevCloseDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowPrevCloseDif] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.OpenPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.OpenPrevCloseDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenPrevCloseDif] > Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeOpenChange) == AnalyzesDataSet.AnalyzeCombination.NegativeOpenChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighChange) == AnalyzesDataSet.AnalyzeCombination.NegativeHighChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeLowChange) == AnalyzesDataSet.AnalyzeCombination.NegativeLowChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeCloseChange) == AnalyzesDataSet.AnalyzeCombination.NegativeCloseChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeVolumeChange) == AnalyzesDataSet.AnalyzeCombination.NegativeVolumeChange)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.VolumeChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighLowDif) == AnalyzesDataSet.AnalyzeCombination.NegativeHighLowDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighLowDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighOpenDif) == AnalyzesDataSet.AnalyzeCombination.NegativeHighOpenDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighOpenDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeLowOpenDif) == AnalyzesDataSet.AnalyzeCombination.NegativeLowOpenDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowOpenDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeCloseOpenDif) == AnalyzesDataSet.AnalyzeCombination.NegativeCloseOpenDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseOpenDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.NegativeHighPrevCloseDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighPrevCloseDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeLowPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.NegativeLowPrevCloseDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowPrevCloseDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
//            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeOpenPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.NegativeOpenPrevCloseDif)
//            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenPrevCloseDif] > -Constants.PredictionErrorRange) ? 1 : 0; }

//            return combinedPrediction;
//        }

//        private string GetDataString(int rowNumber)
//        {
//            string dataString = GetCombinationName(rowNumber);

//            for (int i = 0; i < NumOfDataColumns; i++)
//            {
//                dataString += "," + this[rowNumber * NumOfDataColumns + i].ToString();
//            }

//            return dataString;
//        }

//        private string GetColumnNamesString()
//        {
//            string columnNames = "Combination";

//            for (int depth = Constants.MinDepthRange; depth <= Constants.MaxDepthRange; depth++)
//            {
//                string depthPrefix = "D" + depth.ToString() + "-Up-";
//                for (int changePrediction = 0; changePrediction < Constants.AnalyzeChangesList.Count; changePrediction++)
//                {
//                    columnNames += "," + depthPrefix + Constants.AnalyzeChangesList[changePrediction].ToString();
//                }

//                depthPrefix = "D" + depth.ToString() + "-Down-";
//                for (int changePrediction = 0; changePrediction < Constants.AnalyzeChangesList.Count; changePrediction++)
//                {
//                    columnNames += "," + depthPrefix + Constants.AnalyzeChangesList[changePrediction].ToString();
//                }
//            }

//            return columnNames;
//        }

//        private string GetCombinationName(int rowNumber)
//        {
//            string variationName = m_AnalyzeCombinations[rowNumber].ToString().Replace(", ", "-");

//            return variationName;
//        }

//        #endregion
//    }
//}
