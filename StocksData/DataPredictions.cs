using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class DataPredictions : Dictionary<ulong /*Combination*/, List<double>>
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

        private GPUPredictions m_GPUPredictions = null;

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

        private string m_PredictionDataSetName = string.Empty;

        public string PredictionDataSetName
        {
            get { return m_PredictionDataSetName; }
            set { m_PredictionDataSetName = value; }
        }

        public DataSet DataSet { get; internal set; }
        public double GPULoadTime { get; internal set; }

        public int MinimumChangesForPrediction { get { return (int)(DataSet.NumOfRows * DSSettings.MinimumChangesForPredictionRatio); } }

        #endregion

        #region Constructors

        private DataPredictions(DataPredictions dataPredictions)
        {
            DataSet = dataPredictions.DataSet;
            PredictionDataSetName = dataPredictions.PredictionDataSetName;
        }

        public DataPredictions(string dataPredictionFilePath)
        {
            LoadFromFile(dataPredictionFilePath);
            m_PredictionDataSetName = Path.GetFileNameWithoutExtension(dataPredictionFilePath);
            // CaclulatePredictions();
        }

        public DataPredictions(string dataSetFilePath, string dataPredictionFilePath)
        {
            DataSet = new DataSet(dataSetFilePath);
            LoadFromFile(dataPredictionFilePath);
            m_PredictionDataSetName = Path.GetFileNameWithoutExtension(dataPredictionFilePath);
            // CaclulatePredictions();
        }

        public DataPredictions(string filePath, DataSet dataSet)
        {
            LoadFromFile(filePath);
            DataSet = dataSet;
            m_PredictionDataSetName = Path.GetFileNameWithoutExtension(filePath);
           // CaclulatePredictions();
        }

        public DataPredictions(DataSet dataSet, string predictionFolder, bool useGPU = true)
        {
            DataSet = dataSet;
            m_PredictionDataSetName = DataSet.DataSetName + DSSettings.PredictionSuffix;
            string filePath = predictionFolder + "\\" + PredictionDataSetName + ".csv";

            if (File.Exists(filePath))
            {
                LoadFromFile(filePath, true);
            }

            LoadFromDataSet(useGPU);
        }

        public void AnalyzeChangesEffects()
        {
            Dictionary<CombinationItem, Dictionary<string, string>> changeEffects = new Dictionary<CombinationItem, Dictionary<string, string>>();
            foreach (ulong changeItem in DSSettings.ULongToCombinationItemMap.Keys)
            {
                CombinationItem combinationItem = DSSettings.ULongToCombinationItemMap[changeItem];
                changeEffects.Add(combinationItem, new Dictionary<string, string>());
                double instances = Keys.Where(x => (x & changeItem) == changeItem).Count();
                Console.WriteLine("Num of combinations containing {0} is {1} - {2}%", combinationItem.ToString(), Keys.Where(x => (x & changeItem) == changeItem).Count(), instances, instances / (double)Count * 100);
            }
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
                PredictCPU();
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

            if (loadBadPredictions && File.Exists(filePath.Substring(0, filePath.LastIndexOf('\\')) + "\\" + PredictionDataSetName + "_BadCombinations.csv"))
            {
                using (StreamReader reader = new StreamReader(filePath.Substring(0, filePath.LastIndexOf('\\')) + "\\" + PredictionDataSetName + "_BadCombinations.csv"))
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
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            using (StreamWriter csvFile = new StreamWriter(folderPath + "\\" + PredictionDataSetName + ".csv"))
            {
                // Write the first line
                csvFile.WriteLine(GetColumnNamesString());

                foreach (ulong combination in this.Keys)
                {
                    csvFile.WriteLine(GetDataString(combination));
                }
            }
            using (StreamWriter writer = new StreamWriter(folderPath + "\\" + PredictionDataSetName + "_BadCombinations.csv"))
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

            ulong combination = Convert.ToUInt64(data[0]);

            List<double> combinationPrediction = new List<double>();

            for (int column = 2; column < data.Length; column++)
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
                            CombinationULong = combination,
                            Combination = CombinationItem.ULongToCombinationItems(combination),
                            PredictionCorrectness = this[combination][dataColumn],
                            PredictedChange = DSSettings.PredictionItems[dataColumn],
                            DataSet = DataSet,
                            DataPredictions = this,
                        });
                    }
                }
            }

            return predictionRecords.OrderByDescending(x => x.PredictionCorrectness).ToList();
        }

        #endregion

        #region Private Methods

        private void InitializePredictedCollectionsCPU()
        {
            InitializeOneItemPredictedCollections();

            for (int combinationSize = 2; combinationSize <= DSSettings.PredictionMaxCombinationSize; combinationSize++)
            {
                InitializePredictedCollectionsCPU(combinationSize);
            }
        }

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
                    if (predictions.Count >= MinimumChangesForPrediction)
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

        private void InitializeOneItemPredictedCollections()
        {
            if (DataSet == null)
            {
                return;
            }

            m_PredictedCollections = new Dictionary<ulong, List<int>>();
            foreach (CombinationItem combinationItem in DSSettings.ChangeItems.OrderBy(x => x))
            {
                List<int> combinationPredictions = new List<int>();
                ulong combinationItemULong = combinationItem.ToULong();

                for (int dataRow = 0; dataRow < DataSet.NumOfRows - combinationItem.Range * 3; dataRow++)
                {
                    if (IsContainsPrediction(combinationItem, dataRow + combinationItem.Range, DSSettings.PredictionErrorRange, -DSSettings.PredictionErrorRange))
                    {
                        combinationPredictions.Add(dataRow);
                    }
                }

                if (combinationPredictions.Count >= MinimumChangesForPrediction)
                {
                    m_PredictedCollections.Add(combinationItemULong, combinationPredictions);
                }
            }
        }

        public bool IsContainsPrediction(CombinationItem combinationItem, int dataRow, double upperErrorBorder, double lowerErrorBorder)
        {
            ChangeMap changeMap = DSSettings.DataItemsCalculationMap[combinationItem.DataItem];

            double change = CalculateChange(dataRow, combinationItem.Range, changeMap.FromData, changeMap.OfData, changeMap.FromOffset, changeMap.OfOffset, changeMap.Offset);

            if ((changeMap.IsPositiveChange && change > upperErrorBorder) || (!changeMap.IsPositiveChange && change < lowerErrorBorder))
            {
                return true;
            }
            
            return false;
        }

        public bool IsGoodPrediction(CombinationItem changeItem, CombinationItem predictedItem, int dataRow, double upperErrorBorder, double lowerErrorBorder)
        {
            if (!IsContainsPrediction(changeItem, dataRow + changeItem.Range - 1, upperErrorBorder, lowerErrorBorder))
            {
                return false;
            }

            ChangeMap changeMap = DSSettings.DataItemsCalculationMap[predictedItem.DataItem];

            double fromAverage = CalculateAverage(dataRow, predictedItem.Range - 1, changeMap.FromData);
            double ofAverage = CalculateAverage(dataRow + predictedItem.Range, predictedItem.Range, changeMap.FromData);
            double change = (fromAverage - ofAverage) / ofAverage;

            if ((changeMap.IsPositiveChange && change <= upperErrorBorder) || (!changeMap.IsPositiveChange && change >= lowerErrorBorder))
            {
                return true;
            }

            return false;
        }

        private double CalculateChange(int dataRow, int range, DataSet.DataColumns dataColumFrom, DataSet.DataColumns dataColumOf, int fromRowOffset, int ofRowOffset, int offset)
        {
            int dataFromStartPosition = fromRowOffset * range + offset;
            int dataOfStartPosition = ofRowOffset * range + offset;
            double sumOf = 0;
            double sumFrom = 0;
            for (int i = dataRow; i < dataRow + range; i++)
            {
                sumOf += DataSet[(i + dataOfStartPosition) * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumOf];
                sumFrom += DataSet[(i + dataFromStartPosition) * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumFrom];
            }

            return (sumFrom - sumOf) / sumOf / range;
        }

        private double CalculateAverage(int dataRow, int range, DataSet.DataColumns dataColum)
        {
            double sum = 0;
            for (int i = dataRow; i < dataRow + range; i++)
            {
                sum += DataSet[i * (int)DataSet.DataColumns.NumOfColumns + (int)dataColum];
            }

            return sum / range;
        }

        private void PredictCPU()
        {
            InitializePredictedCollectionsCPU();

            Console.WriteLine();
            int i = 1;
            foreach (ulong combination in m_PredictedCollections.Keys)
            {
                DoPredictionCombination(combination);

                Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.Write(new string(' ', Console.WindowWidth));
                Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Predicted {0}%", (((double)i) / (double)m_PredictedCollections.Count * 100.0).ToString("0.00"));
                i++;
            }
        }

        private void PredictGPU()
        {
            //InitializeOneItemPredictedCollections();

            m_GPUCombinations = new ulong[DSSettings.GPUCycleSize];
            
            DateTime timePoint = DateTime.Now;
            m_GPUPredictions = new GPUPredictions(DataSet.ToArray(), 
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

        public DataPredictions PredictionGPUTest()
        {
            DataPredictions dataPredictionTest = new DataPredictions(this);

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

                m_GPUPredictions = new GPUPredictions(DataSet.ToArray(),
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
                if (i % 2 == 1 && currentCombinationItems.Contains((byte)(i - 1)) && i != 9 && i != 11)
                {
                    continue;
                }
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
                combinationSize, m_GPUCombinationNum, MinimumChangesForPrediction, DSSettings.MinimumRelevantPredictionResult);
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

        internal static List<DataItem> GetDataItems()
        {
            List<DataItem> dataItems = typeof(DataItem).GetEnumValues().Cast<DataItem>().ToList();
            dataItems.Remove(dataItems.First());
            return dataItems;
        }

        private void DoPredictionCombination(ulong combination)
        {
            double[] correctPredictions = new double[DSSettings.PredictionItems.Count];

            for (int i = 0; i < DSSettings.PredictionItems.Count; i++)
            {
                correctPredictions[i] = 0.0;
            }

            for (int i = 0; i < m_PredictedCollections[combination].Count; i++)
            {
                for (int predictionCombinationNum = 0; predictionCombinationNum < DSSettings.PredictionItems.Count; predictionCombinationNum++)
                {
                    if (m_PredictedCollections[combination][i] > DataSet.NumOfRows - DSSettings.PredictionItems[predictionCombinationNum].Range * 2)
                    {
                        break;
                    }
                    if (IsContainsPrediction(DSSettings.PredictionItems[predictionCombinationNum], m_PredictedCollections[combination][i], DSSettings.PredictionErrorRange, -DSSettings.PredictionErrorRange))
                    {
                        correctPredictions[predictionCombinationNum]++;
                    }
                }
            }

            bool containsRelevantResults = false;
            List<double> predictionResults = new double[DSSettings.PredictionItems.Count].ToList();

            for (int i = 0; i < DSSettings.PredictionItems.Count; i++)
            {
                predictionResults[i] = correctPredictions[i] / m_PredictedCollections[combination].Count;
                if (predictionResults[i] > DSSettings.MinimumRelevantPredictionResult)
                {
                    containsRelevantResults = true;
                }
            }

            if (containsRelevantResults)
            {
                Add(combination, predictionResults);
            }
        }

        private string GetDataString(ulong combination)
        {
            string dataString = combination.ToString() + "," + CombinationItem.CombinationToString(combination);

            for (int i = 0; i < NumOfDataColumns; i++)
            {
                dataString += "," + this[combination][i].ToString();
            }

            return dataString;
        }

        private string GetColumnNamesString()
        {
            string columnNames = "Combination,CombinationItems";

            for (int changePrediction = 0; changePrediction < DSSettings.PredictionItems.Count; changePrediction++)
            {
                columnNames += "," + DSSettings.PredictionItems[changePrediction].ToString();
            }

            return columnNames;
        }

        #endregion
    }
}
