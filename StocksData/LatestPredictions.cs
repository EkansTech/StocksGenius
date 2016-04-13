using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class LatestPredictions : Dictionary<ulong /*Combination*/, List<double>>
    {
        #region Members

        private Dictionary<ulong /*combination*/, List<int>> m_PredictedCollections = null;

        private ulong[] m_GPUCombinations = null;
        private byte[] m_GPUCombinationsItems = null;
        private int m_GPUCombinationNum = 0;
        private int m_GPUCyclesPerSize = 0;
        private Dictionary<byte, ulong[]> m_GPUBadCombinations = new Dictionary<byte, ulong[]>();
        private List<ulong> m_GPUSizeBadCombinations = null;
        private ulong m_GPUNumOfBadCombinations = 0;
        private List<int> m_GPUCurrentBadCombination;
        private bool[] m_GPUOnBadSequence = null;
        private ulong[] m_GPUBadSequenceNum = null;

        private GPULatestPredictions m_GPUPredictions = null;

        private byte m_LasPredictedSize = 0;

        #endregion

        #region Properties

        private Dictionary<int /*range*/, List<List<double>>> m_CalculatedPredictions = null;

        public Dictionary<int /*range*/, List<List<double>>> CalculatedPredictions
        {
            get { return m_CalculatedPredictions; }
            set { m_CalculatedPredictions = value; }
        }


        public int NumOfDataColumns = DSSettings.PredictionItems.Count;

        public int NumOfDataRows
        {
            get { return Count / NumOfDataColumns; }
        }

        private string m_PredictionsName = string.Empty;

        public string PredictionsName
        {
            get { return m_PredictionsName; }
            set { m_PredictionsName = value; }
        }

        public List<DataSet> DataSets { get; private set; }
        public double GPULoadTime { get; internal set; }

        #endregion

        #region Constructors

        private LatestPredictions(LatestPredictions latestPredictions)
        {
            DataSets = latestPredictions.DataSets;
            PredictionsName = latestPredictions.PredictionsName;
        }

        public LatestPredictions(List<DataSet> dataSets, string latestPredictionsFilePath)
        {
            DataSets = dataSets;
            LoadFromFile(latestPredictionsFilePath);
            m_PredictionsName = Path.GetFileNameWithoutExtension(latestPredictionsFilePath);
        }

        public LatestPredictions(string dataPredictionFilePath, List<string> dataSetFilePaths)
        {
            LoadFromFile(dataPredictionFilePath);
            DataSets = new List<DataSet>();
            foreach (string dataSetFilePath in dataSetFilePaths)
            {
                DataSets.Add(new DataSet(dataSetFilePath));
            }
            m_PredictionsName = Path.GetFileNameWithoutExtension(dataPredictionFilePath);
        }

        public LatestPredictions(List<string> dataSetFilePaths, string dataPredictionFilePath, bool useGPU = true)
        {
            DataSets = new List<DataSet>();
            foreach (string dataSetFilePath in dataSetFilePaths)
            {
                DataSets.Add(new DataSet(dataSetFilePath, TestDataAction.LoadOnlyPredictionData));
            }
            m_PredictionsName = Path.GetFileNameWithoutExtension(dataPredictionFilePath);

            if (File.Exists(dataPredictionFilePath))
            {
                LoadFromFile(dataPredictionFilePath, true);
            }

            LoadFromDataSet(useGPU);
        }

        #endregion

        #region Interface

        public void LoadFromDataSet(bool useGPU)
        {
            if (useGPU)
            {
                GPULoadTime = 0;
                PredictGPU();
            }
            else
            {
            }
        }

        public void LoadFromFile(string filePath, bool loadBadPredictions = false)
        {
            using (StreamReader csvFile = new StreamReader(filePath))
            {

                // Read the first line and validate correctness of columns in the data file
                csvFile.ReadLine();

                while (!csvFile.EndOfStream)
                {
                    Add(csvFile.ReadLine());
                }
            }
            m_LasPredictedSize = (Count > 0) ? (byte)CombinationItem.ULongToCombinationItems(this.Last().Key).Count : (byte)1;

            if (loadBadPredictions && File.Exists(filePath.Substring(0, filePath.LastIndexOf('\\')) + "\\" + PredictionsName + "_BadCombinations.csv"))
            {
                using (StreamReader reader = new StreamReader(filePath.Substring(0, filePath.LastIndexOf('\\')) + "\\" + PredictionsName + "_BadCombinations.csv"))
                {
                    while (!reader.EndOfStream)
                    {
                        byte size = Convert.ToByte(reader.ReadLine().TrimEnd(new char[] { ':' }));
                        string[] badCombinations = reader.ReadLine().Split(',');
                        m_GPUBadCombinations.Add(size, new ulong[badCombinations.Length]);
                        m_LasPredictedSize = (m_LasPredictedSize > size) ? m_LasPredictedSize : size;
                        for (int i = 0; i < badCombinations.Length; i++)
                        {
                            if (!string.IsNullOrWhiteSpace(badCombinations[i]))
                            {
                                m_GPUBadCombinations[size][i] = Convert.ToUInt64(badCombinations[i]);
                            }
                        }
                    }
                }
            }
        }
        public void SaveDataToFile(string folderPath)
        {
            using (StreamWriter csvFile = new StreamWriter(folderPath + "\\" + PredictionsName + ".csv"))
            {
                // Write the first line
                csvFile.WriteLine(GetColumnNamesString());

                foreach (ulong combination in this.Keys)
                {
                    csvFile.WriteLine(GetDataString(combination));
                }
            }
            using (StreamWriter writer = new StreamWriter(folderPath + "\\" + PredictionsName + "_BadCombinations.csv"))
            {
                foreach (byte size in m_GPUBadCombinations.Keys)
                {
                    if (m_GPUBadCombinations[size].Length > 0)
                    {
                        writer.WriteLine("{0}:", size);
                        writer.Write(m_GPUBadCombinations[size][0].ToString());
                        for (int i = 1; i < m_GPUBadCombinations[size].Length; i++)
                        {
                            writer.Write("," + m_GPUBadCombinations[size][i].ToString());
                        }
                        writer.WriteLine();
                    }
                }
            }
        }

        public void Add(string dataLine)
        {
            string[] data = dataLine.Split(',');

            ulong combination = CombinationItem.StringToCombinationULong(data[0]);

            List<double> combinationPrediction = new List<double>();

            for (int column = 1; column < data.Length; column++)
            {
                combinationPrediction.Add(Convert.ToDouble(data[column]));
            }

            Add(combination, combinationPrediction);
        }

        public List<PredictionRecord> GetBestPredictions(double effectivePredictionResult)
        {
            List<PredictionRecord> predictionRecords = new List<PredictionRecord>();
            foreach (ulong combination in Keys)
            {
                for (int dataColumn = 0; dataColumn < NumOfDataColumns; dataColumn++)
                {
                    if (this[combination][dataColumn] >= effectivePredictionResult)
                    {
                        predictionRecords.Add(new PredictionRecord()
                        {
                            Combination = CombinationItem.ULongToCombinationItems(combination),
                            PredictionCorrectness = this[combination][dataColumn],
                            PredictedChange = DSSettings.PredictionItems[dataColumn],
                        });
                    }
                }
            }

            return predictionRecords.OrderByDescending(x => x.PredictionCorrectness).ToList();
        }

        #endregion

        #region Private Methods

        private void InitializePredictedCollectionsCPU(int combinationSize, ulong combination = 0, int combinationPart = 0, int startPosition = 0)
        {
            for (int i = startPosition; i < DSSettings.ChangeItems.Count - (combinationSize - combinationPart - 1); i++)
            {
                if (!m_PredictedCollections.ContainsKey(DSSettings.ChangeItems[i].ToULong()))
                {
                    continue;
                }

                if (combinationPart == combinationSize - 1)
                {
                    List<int> predictions = CombineLists(m_PredictedCollections[combination], m_PredictedCollections[DSSettings.ChangeItems[i].ToULong()]);
                    if (predictions.Count >= DSSettings.MinimumChangesForPrediction)
                    {
                        m_PredictedCollections.Add(combination | DSSettings.ChangeItems[i].ToULong(), predictions);
                    }
                }
                else
                {
                    combination |= DSSettings.ChangeItems[i].ToULong();

                    if (!m_PredictedCollections.ContainsKey(combination))
                    {
                        continue;
                    }

                    InitializePredictedCollectionsCPU(combinationSize, combination, combinationPart + 1, i + 1);

                    combination &= ~DSSettings.ChangeItems[i].ToULong();
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

        public bool IsContainsPrediction(DataSet dataSet, CombinationItem combinationItem, int dataRow, double upperErrorBorder, double lowerErrorBorder)
        {
            if (combinationItem.DataItem == DataItem.OpenChange
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Open, DataSet.DataColumns.Open, false) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.CloseChange
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Close, DataSet.DataColumns.Close, false) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.VolumeChange
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, false) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.CloseOpenDif
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Close, DataSet.DataColumns.Open, true) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.OpenPrevCloseDif
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Open, DataSet.DataColumns.Close, false) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeOpenChange
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Open, DataSet.DataColumns.Open, false) < lowerErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeCloseChange
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Close, DataSet.DataColumns.Close, false) < lowerErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeVolumeChange
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, false) < lowerErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeCloseOpenDif
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Close, DataSet.DataColumns.Open, true) < lowerErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeOpenPrevCloseDif
                && CalculateChange(dataSet, dataRow, combinationItem.Range, DataSet.DataColumns.Open, DataSet.DataColumns.Close, false) < lowerErrorBorder)
            { return true; }

            return false;
        }

        private double CalculateChange(DataSet dataSet, int dataRow, int range, DataSet.DataColumns dataColumFrom, DataSet.DataColumns dataColumOf, bool isDifFromCurrentDate)
        {
            int dataOfStartPosition = isDifFromCurrentDate ? 0 : range;
            double sumOf = 0;
            double sumFrom = 0;
            for (int i = dataRow; i < dataRow + range; i++)
            {
                sumOf += dataSet[(i + dataOfStartPosition) * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumOf];
                sumFrom += dataSet[i * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumFrom];
            }

            return (sumFrom - sumOf) / sumOf / range;
        }

        private void PredictGPU()
        {
            //InitializeOneItemPredictedCollections();

            m_GPUCombinations = new ulong[DSSettings.GPUCycleSize];
            double[] dataSetsData = DataSets[0].ToArray();
            for (int i = 1; i < DataSets.Count; i++)
            {
                dataSetsData = dataSetsData.Concat(DataSets[i]).ToArray();
            }
            
            DateTime timePoint = DateTime.Now;
            m_GPUPredictions = new GPULatestPredictions(dataSetsData, DataSets.Count, 
                DSSettings.ChangeItems.Select(x => DSSettings.DataItems.IndexOf(x.DataItem)).ToArray(),
                DSSettings.ChangeItems.Select(x => x.Range).ToArray(),
                DSSettings.PredictionItems.Select(x => DSSettings.DataItems.IndexOf(x.DataItem)).ToArray(),
                DSSettings.PredictionItems.Select(x => x.Range).ToArray(),
                DSSettings.GPUCycleSize);
            GPULoadTime += (double)(DateTime.Now - timePoint).TotalMilliseconds;

            m_GPUOnBadSequence = new bool[DSSettings.PredictionMaxCombinationSize + 1];
            m_GPUBadSequenceNum = new ulong[DSSettings.PredictionMaxCombinationSize + 1];

            for (byte combinationSize = (byte)(m_LasPredictedSize + 1); combinationSize <= DSSettings.PredictionMaxCombinationSize; combinationSize++)
            {
                m_GPUSizeBadCombinations = new List<ulong>();
                m_GPUCurrentBadCombination = new List<int>();
                for (int i = 0; i <= combinationSize; i++)
                {
                    m_GPUCurrentBadCombination.Add(0);
                    m_GPUOnBadSequence[i] = false;
                    m_GPUBadSequenceNum[i] = 0;
                }
                m_GPUCyclesPerSize = 0;
                m_GPUCombinationsItems = new byte[DSSettings.GPUCycleSize * combinationSize];
                List<byte> currentCombinationItems = new List<byte>(combinationSize);
                PredictGPU(combinationSize, currentCombinationItems);
                int numOfCombinationsPerSize = DSSettings.GPUCycleSize * m_GPUCyclesPerSize + m_GPUCombinationNum;
                RunGpuPredictionCycle(combinationSize);
                Console.WriteLine("Number of combinations for size {0} is {1}", combinationSize, numOfCombinationsPerSize);
                Console.WriteLine("Number of bad combinations is {0}", m_GPUNumOfBadCombinations);

                if (!m_GPUBadCombinations.ContainsKey(combinationSize))
                {
                    m_GPUBadCombinations.Add(combinationSize, m_GPUSizeBadCombinations.ToArray());
                }
            }

            m_GPUPredictions.FreeGPU();
        }

        public LatestPredictions PredictionGPUTest()
        {
            LatestPredictions dataPredictionTest = new LatestPredictions(this);

            foreach (ulong key in Keys)
            {
                dataPredictionTest.Add(key, null);
            }

            Dictionary<byte, Dictionary<ulong, byte[]>> gpuCombinationsItems = new Dictionary<byte, Dictionary<ulong, byte[]>>();

            foreach (ulong combination in Keys)
            {
                List<CombinationItem> combinationItems = CombinationItem.ULongToCombinationItems(combination);
                if (!gpuCombinationsItems.ContainsKey((byte)combinationItems.Count))
                {
                    gpuCombinationsItems.Add((byte)combinationItems.Count, new Dictionary<ulong, byte[]>());
                }
                gpuCombinationsItems[(byte)combinationItems.Count].Add(combination, combinationItems.Select(x => DSSettings.ChangeItemsMap[x]).ToArray<byte>());
            }

            foreach (byte combinationSize in gpuCombinationsItems.Keys)
            {
                byte[] combinations = new byte[gpuCombinationsItems[combinationSize].Count * combinationSize];
                int i = 0;
                foreach (ulong combination in gpuCombinationsItems[combinationSize].Keys)
                {
                    Array.Copy(gpuCombinationsItems[combinationSize][combination], 0, combinations, i * combinationSize, combinationSize);
                    i++;
                }
                double[] dataSetsData = DataSets[0].ToArray();
                for (int j = 1; j < DataSets.Count; j++)
                {
                    dataSetsData = dataSetsData.Concat(DataSets[j]).ToArray();
                }

                m_GPUPredictions = new GPULatestPredictions(dataSetsData, DataSets.Count,
                    DSSettings.ChangeItems.Select(x => DSSettings.DataItems.IndexOf(x.DataItem)).ToArray(),
                    DSSettings.ChangeItems.Select(x => x.Range).ToArray(),
                    DSSettings.PredictionItems.Select(x => DSSettings.DataItems.IndexOf(x.DataItem)).ToArray(),
                    DSSettings.PredictionItems.Select(x => x.Range).ToArray(),
                    gpuCombinationsItems[combinationSize].Count);

                double[] predictionResultsArray = m_GPUPredictions.PredictCombinationsTest(combinations.ToArray(), combinationSize, gpuCombinationsItems[combinationSize].Count, 0, DSSettings.TestRange);


                m_GPUPredictions.FreeGPU();

                List<double> predictionResults= predictionResultsArray.ToList();

                i = 0;

                foreach (ulong combination in gpuCombinationsItems[combinationSize].Keys)
                {
                    dataPredictionTest[combination] = predictionResults.GetRange(i * NumOfDataColumns, NumOfDataColumns);
                    i++;
                }
            }


            return dataPredictionTest;
        }

        private void PredictGPU(byte combinationSize, List<byte> currentCombinationItems, ulong combination = 0, byte combinationPart = 0, byte startPosition = 0)
        {
            if (combinationPart == combinationSize)
            {
                AddGPUCombination(combination, combinationSize, currentCombinationItems);
                return;
            }

            for (byte i = startPosition; i < DSSettings.ChangeItems.Count - (combinationSize - combinationPart - 1); i++)
            {
                combination |= DSSettings.ChangeItems[i].ToULong();
                currentCombinationItems.Add(i);

                if (m_GPUOnBadSequence[combinationPart + 1])
                {
                    m_GPUBadSequenceNum[combinationPart + 1]++;
                    if (m_GPUBadSequenceNum[combinationPart + 1] >= m_GPUBadCombinations[(byte)(combinationPart + 1)][m_GPUCurrentBadCombination[combinationPart + 1]])
                    {
                        m_GPUOnBadSequence[combinationPart + 1] = false;
                        m_GPUCurrentBadCombination[combinationPart + 1]++;
                    }
                }
                else if (m_GPUBadCombinations.ContainsKey((byte)(combinationPart + 1)) 
                    && m_GPUCurrentBadCombination[combinationPart + 1] < m_GPUBadCombinations[(byte)(combinationPart + 1)].Length 
                    && combination == m_GPUBadCombinations[(byte)(combinationPart + 1)][m_GPUCurrentBadCombination[combinationPart + 1]])
                {
                    m_GPUOnBadSequence[combinationPart + 1] = true;
                    m_GPUCurrentBadCombination[combinationPart + 1]++;
                    m_GPUBadSequenceNum[combinationPart + 1] = 1;
                    if (m_GPUBadCombinations[(byte)(combinationPart + 1)][m_GPUCurrentBadCombination[combinationPart + 1]] == 1)
                    {
                        m_GPUOnBadSequence[combinationPart + 1] = false;
                        m_GPUCurrentBadCombination[combinationPart + 1]++;
                    }
                }
                else
                {
                    PredictGPU(combinationSize, currentCombinationItems, combination, (byte)(combinationPart + 1), (byte)(i + 1));
                }

                combination &= ~DSSettings.ChangeItems[i].ToULong();
                currentCombinationItems.Remove(i);
            }
        }

        private void AddGPUCombination(ulong combination, byte combinationSize, List<byte> currentCombinationItems)
        {
            for (int i = 0; i < combinationSize; i++)
            {
                m_GPUCombinationsItems[m_GPUCombinationNum * combinationSize + i] = currentCombinationItems[i];
            }

            m_GPUCombinations[m_GPUCombinationNum] = combination;
            m_GPUCombinationNum++;

            if (m_GPUCombinationNum == DSSettings.GPUCycleSize)
            {
                RunGpuPredictionCycle(combinationSize);
            }
        }

        private void RunGpuPredictionCycle(byte combinationSize)
        {
            m_GPUCyclesPerSize++;
            DateTime timePoint = DateTime.Now;
            double[] predictionResultsArray = m_GPUPredictions.PredictCombinations(m_GPUCombinationsItems, 
                combinationSize, m_GPUCombinationNum, DSSettings.MinimumChangesForPrediction, DSSettings.MinimumRelevantPredictionResult);
            GPULoadTime += (double)(DateTime.Now - timePoint).TotalMilliseconds;
            Console.WriteLine("{0} seconds for {1} combinations", (DateTime.Now - timePoint).TotalMilliseconds / 1000, m_GPUCombinationNum);

            List<double> predictionResults = predictionResultsArray.ToList();

            bool prevBadCombination = false;
            for (int combinationNum = 0; combinationNum < m_GPUCombinationNum; combinationNum++)
            {
                bool badCombination = true;
                for (int resultNum = 0; resultNum < DSSettings.PredictionItems.Count; resultNum++)
                {
                    if (predictionResults[combinationNum * DSSettings.PredictionItems.Count + resultNum] > DSSettings.MinimumRelevantPredictionResult)
                    {
                        Add(m_GPUCombinations[combinationNum], predictionResults.GetRange(combinationNum * DSSettings.PredictionItems.Count, DSSettings.PredictionItems.Count));
                        badCombination = false;
                        break;
                    }

                    if (predictionResults[combinationNum * DSSettings.PredictionItems.Count + resultNum] > DSSettings.GPUHopelessPredictionLimit)
                    {
                        badCombination = false;
                    }
                }

                if (badCombination)
                {
                    if (prevBadCombination)
                    {
                        m_GPUSizeBadCombinations[m_GPUSizeBadCombinations.Count - 1]++;
                    }
                    else
                    {
                        m_GPUSizeBadCombinations.Add(m_GPUCombinations[combinationNum]);
                        m_GPUSizeBadCombinations.Add(1);
                    }
                    m_GPUNumOfBadCombinations++;
                    prevBadCombination = true;
                }
                else
                {
                    prevBadCombination = false;
                }
            }
            
            m_GPUCombinationNum = 0;
        }

        private string GetDataString(ulong combination)
        {
            string dataString = CombinationItem.CombinationToString(combination);

            for (int i = 0; i < NumOfDataColumns; i++)
            {
                dataString += "," + this[combination][i].ToString();
            }

            return dataString;
        }

        private string GetColumnNamesString()
        {
            string columnNames = "Combination";

            for (int changePrediction = 0; changePrediction < DSSettings.PredictionItems.Count; changePrediction++)
            {
                columnNames += "," + DSSettings.PredictionItems[changePrediction].ToString();
            }

            return columnNames;
        }

        #endregion
    }
}
