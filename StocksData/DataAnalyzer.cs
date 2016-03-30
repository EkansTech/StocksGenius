﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    #region Structs   

    public class AnalyzerRecord
    {
        #region Properties

        public List<CombinationItem> Combination { get; set; }

        public double PredictionCorrectness { get; set; }

        public CombinationItem PredictedChange { get; set; }

        #endregion
    }

    public struct CombinationItem
    {
        private int m_RangeNum;
        public int Range;
        public DataItem DataItem;

        public CombinationItem(int range, DataItem combination)
        {
            m_RangeNum = DataAnalyzer.AnalyzeRanges.IndexOf(range);
            Range = range;
            DataItem = combination;
        }

        public CombinationItem(ulong combinationItem)
        {
            m_RangeNum = DataAnalyzer.AnalyzeRanges.Count - 1;
            for (; m_RangeNum > 0; m_RangeNum--)
            {
                if ((((ulong)1) << (m_RangeNum * 10)) <= combinationItem)
                {
                    break;
                }
            }

            Range = DataAnalyzer.AnalyzeRanges[m_RangeNum];
            DataItem = (DataItem)(combinationItem / (((ulong)1) << (m_RangeNum * 10)));
        }

        public CombinationItem(string combinationItemString)
        {
            string[] stringParts = combinationItemString.Split('-');
            Range = Convert.ToInt32(stringParts[0]);
            m_RangeNum = DataAnalyzer.AnalyzeRanges.IndexOf(Range);
            DataItem = (DataItem)Enum.Parse(typeof(DataItem), stringParts[1]);
        }

        public ulong ToULong()
        {
            return (((ulong)1) << (m_RangeNum * 10)) * (ulong)DataItem;
        }

        static public DataItem ULongItemToDataItem(ulong combinationItem)
        {
            int rangeNum = DataAnalyzer.AnalyzeRanges.Count - 1;
            for (; rangeNum > 0; rangeNum--)
            {
                if ((((ulong)1) << (rangeNum * 10)) <= combinationItem)
                {
                    break;
                }
            }
            
            return (DataItem)(combinationItem / (((ulong)1) << (rangeNum * 10)));
        }

        static public int ULongItemToDataItemNum(ulong combinationItem)
        {
            return DataAnalyzer.DataItems.IndexOf(ULongItemToDataItem(combinationItem));
        }

        static public int ULongItemToRange(ulong combinationItem)
        {
            int rangeNum = DataAnalyzer.AnalyzeRanges.Count - 1;
            for (; rangeNum > 0; rangeNum--)
            {
                if ((((ulong)1) << (rangeNum * 10)) <= combinationItem)
                {
                    break;
                }
            }

            return  DataAnalyzer.AnalyzeRanges[rangeNum];
        }

        static public List<CombinationItem> ULongToCombinationItems(ulong combination)
        {
            List<CombinationItem> combinationItems = new List<CombinationItem>();
            for (ulong combinationItem = 1; combinationItem <= (((ulong)1) << (DataAnalyzer.AnalyzeRanges.Count * 10)); combinationItem *= 2)
            {
                if ((combination & combinationItem) != 0)
                {
                    combinationItems.Add(new CombinationItem(combinationItem));
                }
            }

            return combinationItems;
        }

        static public ulong CombinationItemsToULong(List<CombinationItem> combinationTimes)
        {
            ulong combination = 0;
            foreach (CombinationItem combinationItem in combinationTimes)
            {
                combination |= combinationItem.ToULong();
            }

            return combination;
        }

        static public string CombinationToString(ulong combination)
        {
            if (combination == 0)
            {
                return string.Empty;
            }

            List<CombinationItem> combinationItems = ULongToCombinationItems(combination);

            string combinationString = combinationItems[0].ToString();

            for (int i = 1; i < combinationItems.Count; i++)
            {
                combinationString += "+" + combinationItems[i].ToString();
            }

            return combinationString;
        }

        static public List<CombinationItem> StringToItems(string combinationString)
        {
            string[] combinationItemsStrings = combinationString.Split('+');

            if (combinationItemsStrings.Length == 0)
            {
                return null;
            }

            List<CombinationItem> combinationItems = new List<CombinationItem>();

            foreach (string combinationItemString in combinationItemsStrings)
            {
                combinationItems.Add(new CombinationItem(combinationItemString));
            }

            return combinationItems;
        }

        static public ulong StringToCombinationULong(string combinationString)
        {
            string[] combinationItemsStrings = combinationString.Split('+');

            if (combinationItemsStrings.Length == 0)
            {
                return 0;
            }

            ulong combination = 0;

            foreach (string combinationItemString in combinationItemsStrings)
            {
                combination |= (new CombinationItem(combinationItemString)).ToULong();
            }

            return combination;
        }

        public override string ToString()
        {
            return Range.ToString() + "-" + DataItem.ToString();
        }
    }

    #endregion

    #region Enums
    public enum ChangeType
    {
        Down,
        Up
    }

    [Flags]
    public enum DataItem
    {
        None = 0,
        OpenChange = 1,
        CloseChange = OpenChange * 2,
        VolumeChange = CloseChange * 2,
        CloseOpenDif = VolumeChange * 2,
        OpenPrevCloseDif = CloseOpenDif * 2,
        NegativeOpenChange = OpenPrevCloseDif * 2,
        NegativeCloseChange = NegativeOpenChange * 2,
        NegativeVolumeChange = NegativeCloseChange * 2,
        NegativeCloseOpenDif = NegativeVolumeChange * 2,
        NegativeOpenPrevCloseDif = NegativeCloseOpenDif * 2,
    }

    #endregion
    public class DataAnalyzer : Dictionary<ulong /*Combination*/, List<double>>
    {
        #region Constants

        public static readonly List<int> AnalyzeRanges = new List<int>()
        {
            1,
            5,
            20,
            50,
        };

        public static readonly List<CombinationItem> AnalyzeItems = new List<CombinationItem>()
        {
            new CombinationItem(1, DataItem.CloseOpenDif),
            new CombinationItem(1, DataItem.OpenPrevCloseDif),
            new CombinationItem(1, DataItem.NegativeCloseOpenDif),
            new CombinationItem(1, DataItem.NegativeOpenPrevCloseDif),
            new CombinationItem(5, DataItem.CloseOpenDif),
            new CombinationItem(5, DataItem.OpenPrevCloseDif),
            new CombinationItem(5, DataItem.NegativeCloseOpenDif),
            new CombinationItem(5, DataItem.NegativeOpenPrevCloseDif),
            new CombinationItem(20, DataItem.CloseOpenDif),
            new CombinationItem(20, DataItem.OpenPrevCloseDif),
            new CombinationItem(20, DataItem.NegativeCloseOpenDif),
            new CombinationItem(20, DataItem.NegativeOpenPrevCloseDif),
        };

        public const int AnalyzeMaxCombinationSize = 5;

        public const string AnalyzerDataSetSuffix = "-Analyzer";

        public const string AnalyzerDataSetsDir = "\\Analyzer\\";

        public const double PredictionErrorRange = 0.002;

        public const int MinimumPredictionsForAnalyze = 100;

        public const int TestRange = 100;

        public const double MinimumRelevantAnalyzeResult = 0.6;

        public static readonly int GPUCycleSize = 256 * AnalyzeItems.Count;

        public readonly static List<DataItem> DataItems = GetDataItems();

        public static readonly List<ulong> PredictionItems = GetCombinations();

        #endregion

        #region Members

        private Dictionary<ulong /*combination*/, List<int>> m_PredictedCollections = null;

        private ulong[] m_GPUCombinations = null;
        private int[] m_GPUCombinationsItems = null;
        private int m_GPUCombinationNum = 0;

        private GPUAnalyzer m_GPUAnalyzer = null;

        #endregion

        #region Properties

        private Dictionary<int /*range*/, List<List<double>>> m_CalculatedPredictions = null;

        public Dictionary<int /*range*/, List<List<double>>> CalculatedPredictions
        {
            get { return m_CalculatedPredictions; }
            set { m_CalculatedPredictions = value; }
        }


        public int NumOfDataColumns = AnalyzeItems.Count;

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

        public DataSet DataSet { get; private set; }

        #endregion

        #region Constructors

        public DataAnalyzer(string filePath, DataSet dataSet)
        {
            LoadFromFile(filePath);
            DataSet = dataSet;
            m_AnalyzeDataSetName = Path.GetFileNameWithoutExtension(filePath);
           // CaclulatePredictions();
        }

        public DataAnalyzer(DataSet dataSet, bool useGPU = true)
        {
            DataSet = dataSet;

            DataSet.RemoveRange(0, TestRange * (int)DataSet.DataColumns.NumOfColumns);

            LoadFromDataSet(useGPU);
        }

        #endregion

        #region Interface

        public void LoadFromDataSet(bool useGPU)
        {
            m_AnalyzeDataSetName = DataSet.DataSetName + AnalyzerDataSetSuffix;

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
            using (StreamReader csvFile = new StreamReader(filePath))
            {

                // Read the first line and validate correctness of columns in the data file
                csvFile.ReadLine();

                while (!csvFile.EndOfStream)
                {
                    Add(csvFile.ReadLine());
                }
            }
        }
        public void SaveDataToFile(string folderPath)
        {
            using (StreamWriter csvFile = new StreamWriter(folderPath + "\\" + AnalyzeDataSetName + ".csv"))
            {
                // Write the first line
                csvFile.WriteLine(GetColumnNamesString());

                foreach (ulong combination in this.Keys)
                {
                    csvFile.WriteLine(GetDataString(combination));
                }
            }
        }

        public void Add(string dataLine)
        {
            string[] data = dataLine.Split(',');

            ulong combination = CombinationItem.StringToCombinationULong(data[0]);

            List<double> combinationAnalyze = new List<double>();

            for (int column = 1; column < data.Length; column++)
            {
                combinationAnalyze.Add(Convert.ToDouble(data[column]));
            }

            Add(combination, combinationAnalyze);
        }

        public List<AnalyzerRecord> GetBestPredictions(double effectivePredictionResult)
        {
            List<AnalyzerRecord> analyzerRecords = new List<AnalyzerRecord>();
            foreach (ulong combination in Keys)
            {
                for (int dataColumn = 0; dataColumn < NumOfDataColumns; dataColumn++)
                {
                    if (this[combination][dataColumn] >= effectivePredictionResult)
                    {
                        analyzerRecords.Add(new AnalyzerRecord()
                        {
                            Combination = CombinationItem.ULongToCombinationItems(combination),
                            PredictionCorrectness = this[combination][dataColumn],
                            PredictedChange = AnalyzeItems[dataColumn],
                        });
                    }
                }
            }

            return analyzerRecords.OrderByDescending(x => x.PredictionCorrectness).ToList();
        }

        #endregion

        #region Private Methods

        private void InitializePredictedCollectionsCPU()
        {
            InitializeOneItemPredictedCollections();

            for (int combinationSize = 2; combinationSize <= AnalyzeMaxCombinationSize; combinationSize++)
            {
                InitializePredictedCollectionsCPU(combinationSize);
            }
        }

        private void InitializePredictedCollectionsCPU(int combinationSize, ulong combination = 0, int combinationPart = 0, int startPosition = 0)
        {
            for (int i = startPosition; i < PredictionItems.Count - (combinationSize - combinationPart - 1); i++)
            {
                if (!m_PredictedCollections.ContainsKey(PredictionItems[i]))
                {
                    continue;
                }

                if (combinationPart == combinationSize - 1)
                {
                    List<int> predictions = CombineLists(m_PredictedCollections[combination], m_PredictedCollections[PredictionItems[i]]);
                    if (predictions.Count >= MinimumPredictionsForAnalyze)
                    {
                        m_PredictedCollections.Add(combination | PredictionItems[i], predictions);
                    }
                }
                else
                {
                    combination |= PredictionItems[i];

                    if (!m_PredictedCollections.ContainsKey(combination))
                    {
                        continue;
                    }

                    InitializePredictedCollectionsCPU(combinationSize, combination, combinationPart + 1, i + 1);

                    combination &= ~PredictionItems[i];
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
            foreach (ulong combinationItemULong in PredictionItems.OrderBy(x => x))
            {
                List<int> combinationPredictions = new List<int>();
                CombinationItem combinationItem = new CombinationItem(combinationItemULong);

                for (int dataRow = 0; dataRow < DataSet.NumOfRows - combinationItem.Range * 3; dataRow++)
                {
                    if (IsContainsPrediction(combinationItem, dataRow + combinationItem.Range, -PredictionErrorRange, PredictionErrorRange))
                    {
                        combinationPredictions.Add(dataRow);
                    }
                }

                if (combinationPredictions.Count >= MinimumPredictionsForAnalyze)
                {
                    m_PredictedCollections.Add(combinationItemULong, combinationPredictions);
                }
            }
        }

        public bool IsContainsPrediction(CombinationItem combinationItem, int dataRow, double upperErrorBorder, double lowerErrorBorder)
        {
            if (combinationItem.DataItem == DataItem.OpenChange
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Open, DataSet.DataColumns.Open, false) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.CloseChange
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Close, DataSet.DataColumns.Close, false) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.VolumeChange
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, false) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.CloseOpenDif
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Close, DataSet.DataColumns.Open, true) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.OpenPrevCloseDif
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Open, DataSet.DataColumns.Close, false) > upperErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeOpenChange
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Open, DataSet.DataColumns.Open, false) < lowerErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeCloseChange
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Close, DataSet.DataColumns.Close, false) < lowerErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeVolumeChange
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, false) < lowerErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeCloseOpenDif
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Close, DataSet.DataColumns.Open, true) < lowerErrorBorder)
            { return true; }
            if (combinationItem.DataItem == DataItem.NegativeOpenPrevCloseDif
                && CalculateChange(dataRow, combinationItem.Range, DataSet.DataColumns.Open, DataSet.DataColumns.Close, false) < lowerErrorBorder)
            { return true; }

            return false;
        }

        //private bool IsContainsPrediction(CombinationItem combinationItem, int dataRow, double upperErrorBorder, double lowerErrorBorder)
        //{
        //    if (combinationItem.DataItem == DataItem.OpenChange && m_CalculatedPredictions[combinationItem.Range][dataRow][0] > upperErrorBorder)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.CloseChange && m_CalculatedPredictions[combinationItem.Range][dataRow][1] > upperErrorBorder)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.VolumeChange && m_CalculatedPredictions[combinationItem.Range][dataRow][2] > upperErrorBorder)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.CloseOpenDif && m_CalculatedPredictions[combinationItem.Range][dataRow][3] > upperErrorBorder)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.OpenPrevCloseDif && m_CalculatedPredictions[combinationItem.Range][dataRow][4] > upperErrorBorder)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeOpenChange && m_CalculatedPredictions[combinationItem.Range][dataRow][5] < lowerErrorBorder)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeCloseChange && m_CalculatedPredictions[combinationItem.Range][dataRow][6] < lowerErrorBorder)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeVolumeChange && m_CalculatedPredictions[combinationItem.Range][dataRow][7] < lowerErrorBorder)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeCloseOpenDif && m_CalculatedPredictions[combinationItem.Range][dataRow][8] < lowerErrorBorder)
        //    { return true; }
        //    if (combinationItem.DataItem == DataItem.NegativeOpenPrevCloseDif && m_CalculatedPredictions[combinationItem.Range][dataRow][9] < lowerErrorBorder)
        //    { return true; }

        //    return false;
        //}

        private double CalculateChange(int dataRow, int range, DataSet.DataColumns dataColumFrom, DataSet.DataColumns dataColumOf, bool isDifFromCurrentDate)
        {
            int dataOfStartPosition = isDifFromCurrentDate ? 0 : range;
            double sumOf = 0.0;
            double sumFrom = 0.0;
            for (int i = dataRow; i < dataRow + range; i++)
            {
                sumOf += DataSet[(dataRow + dataOfStartPosition) * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumOf];
                sumFrom += DataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumFrom];
            }

            return (sumFrom - sumOf) / sumOf / range;
        }

        private void AnalyzeCPU()
        {
            InitializePredictedCollectionsCPU();

            Console.WriteLine();
            int i = 1;
            foreach (ulong combination in m_PredictedCollections.Keys)
            {
                DoAnalyzeCombination(combination);

                Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.Write(new string(' ', Console.WindowWidth));
                Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Analyzed {0}%", (((double)i) / (double)m_PredictedCollections.Count * 100.0).ToString("0.00"));
                i++;
            }
        }

        private void AnalyzeGPU()
        {
            InitializeOneItemPredictedCollections();

            m_GPUCombinations = new ulong[GPUCycleSize];

            m_GPUAnalyzer = new GPUAnalyzer(DataSet.ToArray(), 
                PredictionItems.Select(x => CombinationItem.ULongItemToDataItemNum(x)).ToArray(),
                PredictionItems.Select(x => CombinationItem.ULongItemToRange(x)).ToArray(),
                AnalyzeItems.Select(x => DataAnalyzer.DataItems.IndexOf(x.DataItem)).ToArray(),
                AnalyzeItems.Select(x => x.Range).ToArray());

            for (int combinationSize = 1; combinationSize <= AnalyzeMaxCombinationSize; combinationSize++)
            {
                m_GPUCombinationsItems = new int[GPUCycleSize * combinationSize];
                List<int> currentCombinationItems = new List<int>(combinationSize);
                AnalyzeGPU(combinationSize, currentCombinationItems);
                RunGpuAnalyzerCycle(combinationSize);
            }

            m_GPUAnalyzer.FreeGPU();
        }

        private void AnalyzeGPU(int combinationSize, List<int> currentCombinationItems, ulong combination = 0, int combinationPart = 0, int startPosition = 0)
        {
            if (combinationPart == combinationSize)
            {
                AddGPUCombination(combination, combinationSize, currentCombinationItems);
                return;
            }

            for (int i = startPosition; i < PredictionItems.Count - (combinationSize - combinationPart - 1); i++)
            {
                combination |= PredictionItems[i];
                currentCombinationItems.Add(i);

                AnalyzeGPU(combinationSize, currentCombinationItems, combination, combinationPart + 1, i + 1);

                combination &= ~PredictionItems[i];
                currentCombinationItems.Remove(i);
            }
        }

        private void AddGPUCombination(ulong combination, int combinationSize, List<int> currentCombinationItems)
        {
            for (int i = 0; i < combinationSize; i++)
            {
                m_GPUCombinationsItems[m_GPUCombinationNum * combinationSize + i] = currentCombinationItems[i];
            }

            m_GPUCombinations[m_GPUCombinationNum] = combination;
            m_GPUCombinationNum++;

            if (m_GPUCombinationNum == GPUCycleSize)
            {
                RunGpuAnalyzerCycle(combinationSize);
            }
        }

        private void RunGpuAnalyzerCycle(int combinationSize)
        {
            double[] analyzeResultsArray = m_GPUAnalyzer.AnalyzeCombinations(m_GPUCombinationsItems, combinationSize, m_GPUCombinationNum, MinimumPredictionsForAnalyze);

            List<double> analyzeResults = analyzeResultsArray.ToList();

            for (int combinationNum = 0; combinationNum < m_GPUCombinationNum; combinationNum++)
            {
                for (int resultNum = 0; resultNum < AnalyzeItems.Count; resultNum++)
                {
                    if (analyzeResults[combinationNum * AnalyzeItems.Count + resultNum] > MinimumRelevantAnalyzeResult)
                    {
                        Add(m_GPUCombinations[combinationNum], analyzeResults.GetRange(combinationNum * AnalyzeItems.Count, AnalyzeItems.Count));
                        break;
                    }
                }
            }
            
            m_GPUCombinationNum = 0;
        }

        private void CaclulatePredictions()
        {
            m_CalculatedPredictions = new Dictionary<int, List<List<double>>>();
            foreach (int range in AnalyzeRanges)
            {
                List<List<double>> rangePredictions = new List<List<double>>();

                for (int dataRow = 0; dataRow < DataSet.NumOfRows - range * 2; dataRow++)
                {
                    List<double> rowPredictions = new List<double>();
                    rowPredictions.Add(CalculateChange(dataRow, range, DataSet.DataColumns.Open, DataSet.DataColumns.Open, false));
                    rowPredictions.Add(CalculateChange(dataRow, range, DataSet.DataColumns.Close, DataSet.DataColumns.Close, false));
                    rowPredictions.Add(CalculateChange(dataRow, range, DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, false));
                    rowPredictions.Add(CalculateChange(dataRow, range, DataSet.DataColumns.Close, DataSet.DataColumns.Open, true));
                    rowPredictions.Add(CalculateChange(dataRow, range, DataSet.DataColumns.Open, DataSet.DataColumns.Close, false));

                    rangePredictions.Add(rowPredictions);
                }

                m_CalculatedPredictions.Add(range, rangePredictions);
            }
        }

        private static List<DataItem> GetDataItems()
        {
            List<DataItem> dataItems = typeof(DataItem).GetEnumValues().Cast<DataItem>().ToList();
            dataItems.Remove(dataItems.First());
            return dataItems;
        }

        private static List<ulong> GetCombinations()
        {
            List<ulong> combinations = new List<ulong>();
            
            for (int range = 0; range < AnalyzeRanges.Count; range++)
            {
                for (int dataItem = 0; dataItem < DataItems.Count; dataItem++)
                {
                    combinations.Add(new CombinationItem(AnalyzeRanges[range], DataItems[dataItem]).ToULong());
                }
            }

            return combinations;
        }

        private void DoAnalyzeCombination(ulong combination)
        {
            double[] correctPredictions = new double[AnalyzeItems.Count];

            for (int i = 0; i < AnalyzeItems.Count; i++)
            {
                correctPredictions[i] = 0.0;
            }

            for (int i = 0; i < m_PredictedCollections[combination].Count; i++)
            {
                for (int analyzeCombinationNum = 0; analyzeCombinationNum < AnalyzeItems.Count; analyzeCombinationNum++)
                {
                    if (m_PredictedCollections[combination][i] > DataSet.NumOfRows - AnalyzeItems[analyzeCombinationNum].Range * 2)
                    {
                        break;
                    }
                    if (IsContainsPrediction(AnalyzeItems[analyzeCombinationNum], m_PredictedCollections[combination][i], PredictionErrorRange, -PredictionErrorRange))
                    {
                        correctPredictions[analyzeCombinationNum]++;
                    }
                }
            }

            bool containsRelevantResults = false;
            List<double> analyzeResults = new double[AnalyzeItems.Count].ToList();

            for (int i = 0; i < AnalyzeItems.Count; i++)
            {
                analyzeResults[i] = correctPredictions[i] / m_PredictedCollections[combination].Count;
                if (analyzeResults[i] > MinimumRelevantAnalyzeResult)
                {
                    containsRelevantResults = true;
                }
            }

            if (containsRelevantResults)
            {
                Add(combination, analyzeResults);
            }
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

            for (int changePrediction = 0; changePrediction < AnalyzeItems.Count; changePrediction++)
            {
                columnNames += "," + AnalyzeItems[changePrediction].ToString();
            }

            return columnNames;
        }

        #endregion
    }
}
