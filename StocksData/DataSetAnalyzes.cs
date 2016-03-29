using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class AnalyzesDataSet : List<double>
    {
        #region Enums

        [Flags]
        public enum AnalyzeCombination
        {
            None = 0,
            OpenChange = 1,
            HighChange = OpenChange * 2,
            LowChange = HighChange * 2,
            CloseChange = LowChange * 2,
            VolumeChange = CloseChange * 2,
            HighLowDif = VolumeChange * 2,
            HighOpenDif = HighLowDif * 2,
            LowOpenDif = HighOpenDif * 2,
            CloseOpenDif = LowOpenDif * 2,
            HighPrevCloseDif = CloseOpenDif * 2,
            LowPrevCloseDif = HighPrevCloseDif * 2,
            OpenPrevCloseDif = LowPrevCloseDif * 2,
            NegativeOpenChange = OpenPrevCloseDif * 2,
            NegativeHighChange = NegativeOpenChange * 2,
            NegativeLowChange = NegativeHighChange * 2,
            NegativeCloseChange = NegativeLowChange * 2,
            NegativeVolumeChange = NegativeCloseChange * 2,
            NegativeHighLowDif = NegativeVolumeChange * 2,
            NegativeHighOpenDif = NegativeHighLowDif * 2,
            NegativeLowOpenDif = NegativeHighOpenDif * 2,
            NegativeCloseOpenDif = NegativeLowOpenDif * 2,
            NegativeHighPrevCloseDif = NegativeCloseOpenDif * 2,
            NegativeLowPrevCloseDif = NegativeHighPrevCloseDif * 2,
            NegativeOpenPrevCloseDif = NegativeLowPrevCloseDif * 2,
        }

        #endregion

        #region Members

        private Dictionary<int, Dictionary<AnalyzeCombination, List<int>>> m_PredictedCollections = null;

        #endregion

        #region Properties

        public int NumOfDataColumns = Constants.AnalyzeChangesList.Count * 2 + 2;

        public int NumOfDataRows
        {
            get { return Count / NumOfDataColumns; }
        }

        private string m_AnalyzeDataSetName = string.Empty;

        public string AnalyzeDataSetName
        {
            get { return m_AnalyzeDataSetName; }
            set { m_AnalyzeDataSetName = value; }
        }

        public ChangesDataSet ChangesDataSet { get; private set; }

        public PredictionsDataSet PredictionsDataSet { get; private set; }

        public static readonly List<AnalyzeCombination> CombinationItems = GetCombinationItems();

        #endregion

        #region Constructors

        public AnalyzesDataSet()
        {
        }

        public AnalyzesDataSet(string filePath)
        {
            LoadFromFile(filePath);
            m_AnalyzeDataSetName = Path.GetFileNameWithoutExtension(filePath);
        }

        public AnalyzesDataSet(ChangesDataSet changesDataSet, PredictionsDataSet predictionsDataSet, bool useGPU = true)
        {
            ChangesDataSet = changesDataSet;
            PredictionsDataSet = predictionsDataSet;

            LoadFromPredictionsDataSet(useGPU);
        }

        #endregion

        #region Interface

        public void LoadFromPredictionsDataSet(bool useGPU)
        {
            m_AnalyzeDataSetName = PredictionsDataSet.PredictionDataSetName.Substring(0, PredictionsDataSet.PredictionDataSetName.IndexOf(Constants.PredictionDataSetSuffix)) + Constants.AnalyzeDataSetSuffix;

            if (useGPU)
            {
                AnalyzeGPU();
            }
            else
            {
                AnalyzeCPU();
            }
        }

        public void LoadFromFile(string filePath)
        {
            StreamReader csvFile = new StreamReader(filePath);

            // Read the first line and validate correctness of columns in the data file
            csvFile.ReadLine();

            while (!csvFile.EndOfStream)
            {
                Add(csvFile.ReadLine());
            }
        }
        public void SaveDataToFile(string folderPath)
        {
            using (StreamWriter csvFile = new StreamWriter(folderPath + "\\" + AnalyzeDataSetName + ".csv"))
            {
                // Write the first line
                csvFile.WriteLine(GetColumnNamesString());

                for (int currentRow = 0; currentRow < NumOfDataRows; currentRow++)
                {
                    csvFile.WriteLine(GetDataString(currentRow));
                }
            }
        }

        public void Add(string dataLine)
        {
            string[] data = dataLine.Split(',');

            Add(Convert.ToDouble(data[0]));

            Add((double)(AnalyzeCombination)Enum.Parse(typeof(AnalyzeCombination), data[1].Replace("-", ", ")));

            for (int column = 2; column < data.Length; column++)
            {
                Add(Convert.ToDouble(data[column]));
            }
        }

        public List<AnalyzeRecord> GetBestPredictions(double effectivePredictionResult)
        {
            List<AnalyzeRecord> analyzeRecords = new List<AnalyzeRecord>();
            for (int rowNumber = 0; rowNumber < NumOfDataRows; rowNumber++)
            {
                for (int dataColumn = 2; dataColumn < NumOfDataColumns; dataColumn++)
                {
                    if (this[rowNumber * NumOfDataColumns + dataColumn] >= effectivePredictionResult)
                    {
                        analyzeRecords.Add(new AnalyzeRecord()
                        {
                            Combination = (AnalyzeCombination)this[rowNumber * NumOfDataColumns + 1],
                            Depth = (int)this[rowNumber * NumOfDataColumns],
                            PredictionCorrectness = this[rowNumber * NumOfDataColumns + dataColumn],
                            AnalyzedChange = (dataColumn > Constants.AnalyzeChangesList.Count + 1) ? AnalyzedChange.Down : AnalyzedChange.Up,
                            PredictedChange = (dataColumn > Constants.AnalyzeChangesList.Count + 1) ? Constants.AnalyzeChangesList[dataColumn - 2 - Constants.AnalyzeChangesList.Count] : Constants.AnalyzeChangesList[dataColumn - 2],
                        });
                    }
                }
            }

            return analyzeRecords.OrderByDescending(x => x.PredictionCorrectness).ToList();
        }

        #endregion

        #region Private Methods

        private void InitializePredictedCollections()
        {
            InitializeOneItemPredictedCollections();

            for (int combinationSize = 2; combinationSize <= Constants.AnalyzeMaxCombinationSize; combinationSize++)
            {
                for (int depth = Constants.MinDepthRange; depth <= Constants.MaxDepthRange; depth++)
                {
                    InitializePredictedCollections(depth, combinationSize);
                }
            }
        }

        private void InitializePredictedCollections(int depth, int combinationSize,
            AnalyzeCombination currentCombination = AnalyzeCombination.None, int combinationPart = 0, int startPosition = 0)
        {
            for (int i = startPosition; i < CombinationItems.Count - (combinationSize - combinationPart - 1); i++)
            {
                if (!m_PredictedCollections[depth].ContainsKey(CombinationItems[i]))
                {
                    continue;
                }
                if (combinationPart == combinationSize - 1)
                {
                    List<int> predictions = CombineLists(m_PredictedCollections[depth][currentCombination], m_PredictedCollections[depth][CombinationItems[i]]);
                    if (predictions.Count < Constants.MinimumPredictionForAnalyze)
                    {
                        continue;
                    }

                    m_PredictedCollections[depth].Add(currentCombination | CombinationItems[i], predictions);
                }
                else
                {
                    currentCombination |= CombinationItems[i];

                    if (!m_PredictedCollections[depth].ContainsKey(currentCombination))
                    {
                        continue;
                    }

                    InitializePredictedCollections(depth, combinationSize, currentCombination, combinationPart + 1, i + 1);

                    currentCombination &= ~CombinationItems[i];
                }
            }

        }

        private List<int> CombineLists(List<int> list1, List<int> list2)
        {
            List<int> combinedList = new List<int>();

            if (list1.Count == 0 || list2.Count == 0)
            {
                return combinedList;
            }

            int list1Pos = 0;
            int list2Pos = 0;
            while (list1Pos < list1.Count)
            {
                while (list1[list1Pos] > list2[list2Pos])
                {
                    list2Pos++;

                    if (list2Pos == list2.Count)
                    {
                        return combinedList;
                    }
                }

                if (list1[list1Pos] == list2[list2Pos])
                {
                    combinedList.Add(list1[list1Pos]);
                    list1Pos++;

                    if (list1Pos == list1.Count)
                    {
                        return combinedList;
                    }
                }

                while (list1[list1Pos] < list2[list2Pos])
                {
                    list1Pos++;

                    if (list1Pos == list1.Count)
                    {
                        return combinedList;
                    }
                }
            }

            return combinedList;
        }

        private void InitializeOneItemPredictedCollections()
        {
            if (PredictionsDataSet == null)
            {
                return;
            }

            int predictionRow = 0;
            m_PredictedCollections = new Dictionary<int, Dictionary<AnalyzeCombination, List<int>>>();
            for (int depth = Constants.MinDepthRange; depth <= Constants.MaxDepthRange; depth++)
            {
                m_PredictedCollections.Add(depth, new Dictionary<AnalyzeCombination, List<int>>());

                foreach (AnalyzeCombination combinationItem in CombinationItems)
                {
                    m_PredictedCollections[depth].Add(combinationItem, new List<int>());
                }

                while (predictionRow < PredictionsDataSet.NumOfRows && PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.Depth] == depth)
                {
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenChange] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.OpenChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighChange] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.HighChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowChange] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.LowChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseChange] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.CloseChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.VolumeChange] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.VolumeChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighLowDif] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.HighLowDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighOpenDif] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.HighOpenDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowOpenDif] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.LowOpenDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseOpenDif] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.CloseOpenDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighPrevCloseDif] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.HighPrevCloseDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowPrevCloseDif] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.LowPrevCloseDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenPrevCloseDif] > -Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.OpenPrevCloseDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenChange] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeOpenChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighChange] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeHighChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowChange] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeLowChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseChange] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeCloseChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.VolumeChange] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeVolumeChange].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighLowDif] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeHighLowDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighOpenDif] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeHighOpenDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowOpenDif] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeLowOpenDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseOpenDif] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeCloseOpenDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighPrevCloseDif] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeHighPrevCloseDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowPrevCloseDif] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeLowPrevCloseDif].Add(predictionRow); }
                    if (PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenPrevCloseDif] < Constants.PredictionErrorRange)
                    { m_PredictedCollections[depth][AnalyzeCombination.NegativeOpenPrevCloseDif].Add(predictionRow); }

                    predictionRow++;
                }

                List<AnalyzeCombination> combinationsToRemove = new List<AnalyzeCombination>();
                foreach (AnalyzeCombination combination in m_PredictedCollections[depth].Keys)
                {
                    if (m_PredictedCollections[depth][combination].Count < Constants.MinimumPredictionForAnalyze)
                    {
                        combinationsToRemove.Add(combination);
                    }
                }

                foreach (AnalyzeCombination combination in combinationsToRemove)
                {
                    m_PredictedCollections[depth].Remove(combination);
                }
            }
        }

        private void AnalyzeCPU()
        {
            InitializePredictedCollections();
            for (int depth = Constants.MinDepthRange; depth < Constants.MaxDepthRange; depth++)
            {
                foreach (AnalyzeCombination combination in m_PredictedCollections[depth].Keys)
                {
                    Add(depth);
                    Add((double)combination);
                    DoAnalyzeCombination(depth, combination);
                }
            }
        }

        private void AnalyzeGPU()
        {
            // AddRange(GPUAnalyze.AnalyzePredictions(ChangesDataSet.ToArray(), PredictionsDataSet.ToArray(), m_AnalyzeCombinations.ToArray()));
        }

        private static List<AnalyzeCombination> GetCombinationItems()
        {
            List<AnalyzeCombination> combinationItems = typeof(AnalyzeCombination).GetEnumValues().Cast<AnalyzeCombination>().ToList();

            combinationItems.Remove(combinationItems.First());

            return combinationItems;
        }

        private void DoAnalyzeCombination(int depth, AnalyzeCombination analyzeCombination)
        {
            int depthStartRow = GetDepthStartRow(depth);
            double[] upCorrectPredictions = new double[Constants.AnalyzeChangesList.Count];
            double[] downCorrectPredictions = new double[Constants.AnalyzeChangesList.Count];

            for (int i = 0; i < Constants.AnalyzeChangesList.Count; i++)
            {
                upCorrectPredictions[i] = 0.0;
                downCorrectPredictions[i] = 0.0;
            }

            for (int i = 0; i < m_PredictedCollections[depth][analyzeCombination].Count; i++)
            {
                for (int predictedChangeNum = 0; predictedChangeNum < Constants.AnalyzeChangesList.Count; predictedChangeNum++)
                {
                    double actualChange = ChangesDataSet[(m_PredictedCollections[depth][analyzeCombination][i] - depthStartRow) * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

                    if (actualChange > Constants.PredictionErrorRange)
                    {
                        upCorrectPredictions[predictedChangeNum]++;
                    }
                    if (actualChange < -Constants.PredictionErrorRange)
                    {
                        downCorrectPredictions[predictedChangeNum]++;
                    }
                }
            }

            for (int i = 0; i < Constants.AnalyzeChangesList.Count; i++)
            {
                Add(upCorrectPredictions[i] / m_PredictedCollections[depth][analyzeCombination].Count);
            }

            for (int i = 0; i < Constants.AnalyzeChangesList.Count; i++)
            {
                Add(downCorrectPredictions[i] / m_PredictedCollections[depth][analyzeCombination].Count);
            }

            //int predictionRow = 0;
            //while (predictionRow < PredictionsDataSet.NumOfRows)
            //{
            //     AnalyzePredictionDepth(ref predictionRow, analyzeCombination);
            //}
        }

        private int GetDepthStartRow(int depth)
        {
            int depthStartRow = 0;
            for (int i = Constants.MinDepthRange; i < depth; i++)
            {
                depthStartRow += ChangesDataSet.NumOfRows - i;
            }

            return depthStartRow;
        }

        private void AnalyzePredictionDepth(ref int predictionRow, AnalyzeCombination analyzeCombination)
        {
            double[] upCorrectPredictions = new double[Constants.AnalyzeChangesList.Count];
            double[] downCorrectPredictions = new double[Constants.AnalyzeChangesList.Count];
            double[] predictions = new double[Constants.AnalyzeChangesList.Count];

            double[] upPredictionsResults = new double[Constants.AnalyzeChangesList.Count];
            double[] downPredictionsResults = new double[Constants.AnalyzeChangesList.Count];

            double[] relevantHistoriesNum = new double[Constants.AnalyzeChangesList.Count];

            double currentDepth = PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns];
            int depthStartRow = predictionRow;
            int changesRow = 0;

            for (int i = 0; i < Constants.AnalyzeChangesList.Count; i++)
            {
                upCorrectPredictions[i] = 0.0;
                downCorrectPredictions[i] = 0.0;
                predictions[i] = 0.0;
            }

            for (; predictionRow < PredictionsDataSet.NumOfRows && predictionRow < depthStartRow + Constants.RelevantHistory; predictionRow++)
            {
                for (int predictedChangeNum = 0; predictedChangeNum < Constants.AnalyzeChangesList.Count; predictedChangeNum++)
                {
                    double prediction = GetPredictionResult(predictionRow, analyzeCombination);
                    double actualChange = ChangesDataSet[changesRow * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

                    if (prediction > 0.0)
                    {
                        predictions[predictedChangeNum]++;

                        if (actualChange > Constants.PredictionErrorRange)
                        {
                            upCorrectPredictions[predictedChangeNum]++;
                        }
                        if (actualChange < -Constants.PredictionErrorRange)
                        {
                            downCorrectPredictions[predictedChangeNum]++;
                        }
                    }
                }

                changesRow++;
            }

            for (int i = 0; i < Constants.AnalyzeChangesList.Count; i++)
            {
                if (predictions[i] > 0)
                {
                    upPredictionsResults[i] = upCorrectPredictions[i] / predictions[i];
                    downPredictionsResults[i] = downCorrectPredictions[i] / predictions[i];
                    relevantHistoriesNum[i] = 2;
                }
                else
                {
                    relevantHistoriesNum[i] = 0;
                }
            }

            for (; predictionRow < PredictionsDataSet.NumOfRows && PredictionsDataSet[predictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns] == currentDepth; predictionRow++)
            {
                for (int predictedChangeNum = 0; predictedChangeNum < Constants.AnalyzeChangesList.Count; predictedChangeNum++)
                {
                    double prediction = GetPredictionResult(predictionRow - Constants.RelevantHistory, analyzeCombination);
                    double actualChange = ChangesDataSet[(changesRow - Constants.RelevantHistory) * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

                    if (prediction > 0.0)
                    {
                        predictions[predictedChangeNum]--;

                        if (actualChange > Constants.PredictionErrorRange)
                        {
                            upCorrectPredictions[predictedChangeNum]--;
                        }
                        if (actualChange < -Constants.PredictionErrorRange)
                        {
                            downCorrectPredictions[predictedChangeNum]--;
                        }
                    }

                    prediction = GetPredictionResult(predictionRow, analyzeCombination);
                    actualChange = ChangesDataSet[changesRow * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

                    if (prediction > 0.0)
                    {
                        predictions[predictedChangeNum]++;

                        if (actualChange > Constants.PredictionErrorRange)
                        {
                            upCorrectPredictions[predictedChangeNum]++;
                        }
                        if (actualChange < -Constants.PredictionErrorRange)
                        {
                            downCorrectPredictions[predictedChangeNum]++;
                        }
                    }

                    if (predictions[predictedChangeNum] > 0)
                    {
                        if (relevantHistoriesNum[predictedChangeNum] > 0)
                        {
                            upPredictionsResults[predictedChangeNum] += (upCorrectPredictions[predictedChangeNum] / predictions[predictedChangeNum] - upPredictionsResults[predictedChangeNum]) / relevantHistoriesNum[predictedChangeNum];
                            downPredictionsResults[predictedChangeNum] += (downCorrectPredictions[predictedChangeNum] / predictions[predictedChangeNum] - downPredictionsResults[predictedChangeNum]) / relevantHistoriesNum[predictedChangeNum];
                            relevantHistoriesNum[predictedChangeNum]++;
                        }
                        else
                        {
                            upPredictionsResults[predictedChangeNum] = upCorrectPredictions[predictedChangeNum] / predictions[predictedChangeNum];
                            downPredictionsResults[predictedChangeNum] = downCorrectPredictions[predictedChangeNum] / predictions[predictedChangeNum];
                            relevantHistoriesNum[predictedChangeNum] = 2;
                        }
                    }
                }

                changesRow++;
            }

            AddRange(upPredictionsResults);
            AddRange(downPredictionsResults);
        }

        private double GetPredictionResult(int predictionsRow, AnalyzeCombination analyzeCombination)
        {
            double combinedPrediction = 1.0;
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.OpenChange) == AnalyzesDataSet.AnalyzeCombination.OpenChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighChange) == AnalyzesDataSet.AnalyzeCombination.HighChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.LowChange) == AnalyzesDataSet.AnalyzeCombination.LowChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.CloseChange) == AnalyzesDataSet.AnalyzeCombination.CloseChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.VolumeChange) == AnalyzesDataSet.AnalyzeCombination.VolumeChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.VolumeChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighLowDif) == AnalyzesDataSet.AnalyzeCombination.HighLowDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighLowDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighOpenDif) == AnalyzesDataSet.AnalyzeCombination.HighOpenDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighOpenDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.LowOpenDif) == AnalyzesDataSet.AnalyzeCombination.LowOpenDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowOpenDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.CloseOpenDif) == AnalyzesDataSet.AnalyzeCombination.CloseOpenDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseOpenDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.HighPrevCloseDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighPrevCloseDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.LowPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.LowPrevCloseDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowPrevCloseDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.OpenPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.OpenPrevCloseDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenPrevCloseDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeOpenChange) == AnalyzesDataSet.AnalyzeCombination.NegativeOpenChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighChange) == AnalyzesDataSet.AnalyzeCombination.NegativeHighChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeLowChange) == AnalyzesDataSet.AnalyzeCombination.NegativeLowChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeCloseChange) == AnalyzesDataSet.AnalyzeCombination.NegativeCloseChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeVolumeChange) == AnalyzesDataSet.AnalyzeCombination.NegativeVolumeChange)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.VolumeChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighLowDif) == AnalyzesDataSet.AnalyzeCombination.NegativeHighLowDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighLowDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighOpenDif) == AnalyzesDataSet.AnalyzeCombination.NegativeHighOpenDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighOpenDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeLowOpenDif) == AnalyzesDataSet.AnalyzeCombination.NegativeLowOpenDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowOpenDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeCloseOpenDif) == AnalyzesDataSet.AnalyzeCombination.NegativeCloseOpenDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseOpenDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.NegativeHighPrevCloseDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighPrevCloseDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeLowPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.NegativeLowPrevCloseDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowPrevCloseDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeOpenPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.NegativeOpenPrevCloseDif)
            { combinedPrediction *= (PredictionsDataSet[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenPrevCloseDif] > Constants.PredictionErrorRange) ? 1 : 0; }

            return combinedPrediction;
        }

        private string GetDataString(int rowNumber)
        {
            string dataString = this[rowNumber * NumOfDataColumns].ToString() + "," + ((AnalyzeCombination)this[rowNumber * NumOfDataColumns + 1]).ToString().Replace(", ", "-");

            for (int i = 2; i < NumOfDataColumns; i++)
            {
                dataString += "," + this[rowNumber * NumOfDataColumns + i].ToString();
            }

            return dataString;
        }

        private string GetColumnNamesString()
        {
            string columnNames = "Depth,Combination";

            string prefix = "Up-";
            for (int changePrediction = 0; changePrediction < Constants.AnalyzeChangesList.Count; changePrediction++)
            {
                columnNames += "," + prefix + Constants.AnalyzeChangesList[changePrediction].ToString();
            }

            prefix = "Down-";
            for (int changePrediction = 0; changePrediction < Constants.AnalyzeChangesList.Count; changePrediction++)
            {
                columnNames += "," + prefix + Constants.AnalyzeChangesList[changePrediction].ToString();
            }

            return columnNames;
        }

        #endregion
    }
}
