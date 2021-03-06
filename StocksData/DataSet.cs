﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class DataSet : List<double>
    {
        #region Enums

        public enum DataColumns
        {
            Date,
            Open,
            High,
            Low,
            Close,
            Volume,

            NumOfColumns,
        }

        #endregion

        #region Members

        static Dictionary<string, int> m_ColumnNamesMap = new Dictionary<string, int>()
        {
            { "date", (int)DataColumns.Date },
            { "open", (int)DataColumns.Open },
            { "openingprice", (int)DataColumns.Open },
            { "high", (int)DataColumns.High },
            { "dailyhigh", (int)DataColumns.High },
            { "low", (int)DataColumns.Low },
            { "dailylow", (int)DataColumns.Low },
            { "close", (int)DataColumns.Close },
            { "closingprice", (int)DataColumns.Close },
            { "volume", (int)DataColumns.Volume },
            { "volume(pcs.)", (int)DataColumns.Volume },
        };

        Dictionary<int, int> m_ColumnsMap = new Dictionary<int, int>();

        Dictionary<DateTime, int> m_DateMap = new Dictionary<DateTime, int>();

        #endregion

        #region Properties

        public int NumOfRows
        {
            get { return Count / (int)DataColumns.NumOfColumns; }
        }


        private string m_DataSetCode = string.Empty;

        public string DataSetCode
        {
            get { return m_DataSetCode; }
            set { m_DataSetCode = value; }
        }
        
        #endregion

        #region Constructor

        public DataSet()
        {
        }

        public DataSet(string dataSetCode)
        {
            m_DataSetCode = dataSetCode;
        }

        public DataSet(string dataSetCode, string filePath, TestDataAction testDataAction = TestDataAction.None, DateTime dateUpTo = default(DateTime), int relevantX = 60, TimeType timeType = TimeType.Month)
        {
            LoadDataFromFile(filePath);
            m_DataSetCode = dataSetCode;

            int cellNum = 0;
            long sinceTicks = DSSettings.DataRelevantSince.Ticks;
            for (cellNum = 0; cellNum < Count; cellNum += (int)DataColumns.NumOfColumns)
            {
                if (sinceTicks > this[cellNum])
                {
                    break;
                }
            }

            if (cellNum < Count)
            {
                RemoveRange(cellNum, Count - cellNum);
            }

            switch (testDataAction)
            {
                case TestDataAction.None:
                    break;
                case TestDataAction.LoadLimitedPredictionData:
                    if (Count >= DSSettings.TestRange * (int)DataColumns.NumOfColumns)
                    {
                        RemoveRange(0, DSSettings.TestRange * (int)DataColumns.NumOfColumns);
                    }
                    else
                    {
                        Clear();
                    }
                    if (Count >= DSSettings.DataSetForPredictionsSize * (int)DataColumns.NumOfColumns)
                    {
                        RemoveRange(DSSettings.DataSetForPredictionsSize * (int)DataColumns.NumOfColumns, Count - DSSettings.DataSetForPredictionsSize * (int)DataColumns.NumOfColumns);
                    }
                    break;
                case TestDataAction.LoadOnlyTestData:
                    if (Count > DSSettings.TestMinSize * (int)DataColumns.NumOfColumns)
                    {
                        RemoveRange(DSSettings.TestMinSize * (int)DataColumns.NumOfColumns, Count - DSSettings.TestMinSize * (int)DataColumns.NumOfColumns);
                    }
                    break;
                case TestDataAction.LoadWithoutTestData:
                    if (Count >= DSSettings.TestRange * (int)DataColumns.NumOfColumns)
                    {
                        RemoveRange(0, DSSettings.TestRange * (int)DataColumns.NumOfColumns);
                    }
                    else
                    {
                        Clear();
                    }
                    break;
                case TestDataAction.LoadDataUpTo:
                    int rowNum = GetDayNum(dateUpTo);
                    if (rowNum == -1)
                    {
                        Clear();
                    }
                    else
                    {
                        RemoveRange(0, rowNum * (int)DataColumns.NumOfColumns);
                    }
                    break;
                case TestDataAction.LoadRelevantMonths:
                    rowNum = GetDayNum(dateUpTo);
                    if (rowNum == -1)
                    {
                        Clear();
                    }
                    else
                    {
                        RemoveRange(0, rowNum * (int)DataColumns.NumOfColumns);
                        switch (timeType)
                        {
                            case TimeType.Day:
                                rowNum = GetDayNum(dateUpTo.AddMonths(-relevantX));
                                break;
                            case TimeType.Week:
                                rowNum = GetDayNum(dateUpTo) + relevantX * 5;
                                break;
                            case TimeType.Month:
                                rowNum = GetDayNum(dateUpTo) + relevantX;
                                break;
                            case TimeType.Year:
                                rowNum = GetDayNum(dateUpTo.AddYears(-relevantX));
                                break;
                        }

                        if (rowNum == -1 || rowNum * (int)DataColumns.NumOfColumns > Count)
                        {
                            Clear();
                        }
                        else
                        {
                            RemoveRange(rowNum * (int)DataColumns.NumOfColumns, Count - rowNum * (int)DataColumns.NumOfColumns);
                        }
                    }
                    break;
            }

            MapDates();
        }

        public DataSet(DataSet dataSet, TestDataAction testDataAction = TestDataAction.None, DateTime dateUpTo = default(DateTime), int relevantX = 60, TimeType timeType = TimeType.Month)
        {
            m_DataSetCode = dataSet.DataSetCode;
            AddRange(dataSet);

            int cellNum = 0;
            long sinceTicks = DSSettings.DataRelevantSince.Ticks;
            for (cellNum = 0; cellNum < Count; cellNum += (int)DataColumns.NumOfColumns)
            {
                if (sinceTicks > this[cellNum])
                {
                    break;
                }
            }

            if (cellNum < Count)
            {
                RemoveRange(cellNum, Count - cellNum);
            }

            switch (testDataAction)
            {
                case TestDataAction.None:
                    break;
                case TestDataAction.LoadLimitedPredictionData:
                    if (Count >= DSSettings.TestRange * (int)DataColumns.NumOfColumns)
                    {
                        RemoveRange(0, DSSettings.TestRange * (int)DataColumns.NumOfColumns);
                    }
                    else
                    {
                        Clear();
                    }
                    if (Count >= DSSettings.DataSetForPredictionsSize * (int)DataColumns.NumOfColumns)
                    {
                        RemoveRange(DSSettings.DataSetForPredictionsSize * (int)DataColumns.NumOfColumns, Count - DSSettings.DataSetForPredictionsSize * (int)DataColumns.NumOfColumns);
                    }
                    break;
                case TestDataAction.LoadOnlyTestData:
                    if (Count > DSSettings.TestMinSize * (int)DataColumns.NumOfColumns)
                    {
                        RemoveRange(DSSettings.TestMinSize * (int)DataColumns.NumOfColumns, Count - DSSettings.TestMinSize * (int)DataColumns.NumOfColumns);
                    }
                    break;
                case TestDataAction.LoadWithoutTestData:
                    if (Count >= DSSettings.TestRange * (int)DataColumns.NumOfColumns)
                    {
                        RemoveRange(0, DSSettings.TestRange * (int)DataColumns.NumOfColumns);
                    }
                    else
                    {
                        Clear();
                    }
                    break;
                case TestDataAction.LoadDataUpTo:
                    int rowNum = GetDayNum(dateUpTo);
                    if (rowNum == -1)
                    {
                        Clear();
                    }
                    else
                    {
                        RemoveRange(0, rowNum * (int)DataColumns.NumOfColumns);
                    }
                    break;
                case TestDataAction.LoadRelevantMonths:
                    rowNum = GetDayNum(dateUpTo);
                    if (rowNum == -1)
                    {
                        Clear();
                    }
                    else
                    {
                        RemoveRange(0, rowNum * (int)DataColumns.NumOfColumns);
                        switch (timeType)
                        {
                            case TimeType.Day:
                                rowNum = GetDayNum(dateUpTo.AddMonths(-relevantX));
                                break;
                            case TimeType.Week:
                                rowNum = GetDayNum(dateUpTo) + relevantX * 5;
                                break;
                            case TimeType.Month:
                                rowNum = GetDayNum(dateUpTo) + relevantX;
                                break;
                            case TimeType.Year:
                                rowNum = GetDayNum(dateUpTo.AddYears(-relevantX));
                                break;
                        }

                        if (rowNum == -1)
                        {
                            Clear();
                        }
                        else if (rowNum * (int)DataColumns.NumOfColumns <= Count)
                        {
                            RemoveRange(rowNum * (int)DataColumns.NumOfColumns, Count - rowNum * (int)DataColumns.NumOfColumns);
                        }
                    }
                    break;
            }

            MapDates();
        }

        #endregion

        #region Interface

        public new double this[int index]
        {
            get
            {
                switch (index)
                {
                    case (int)DataColumns.High: return base[index]; 
                    case (int)DataColumns.Close: return base[index];
                    case (int)DataColumns.Low: return base[index];
                    case (int)DataColumns.Volume: return base[index];
                    default:
                        return base[index];                        
                }
            }
            set
            {
                base[index] = value;
            }
        }

        public DateTime GetDate(int rowNum)
        {
            return new DateTime((long)GetData(rowNum, DataColumns.Date));
        }

        public int GetDayNum(DateTime date)
        {
            if (m_DateMap.ContainsKey(date))
            {
                return m_DateMap[date];
            }

            MapDates();

            if (date.Ticks > this[(int)DataColumns.Date])
            {
                return 0;
            }

            return -1;
        }

        public List<double> GetDayData()
        {
            return GetDayData(0);
        }

        public List<double> GetDayData(DateTime day)
        {
            return GetDayData(GetDayNum(day));
        }

        public List<double> GetDayData(int rowNumber)
        {
            return GetRange(rowNumber * (int)DataColumns.NumOfColumns, (int)DataColumns.NumOfColumns);
        }

        public double GetData(int rowNumber, DataColumns dataColumn)
        {
            return this[rowNumber * (int)DataColumns.NumOfColumns + (int)dataColumn];
        }

        public double GetLastChange(int rowNumber, DataColumns dataColumn)
        {
            return (this[rowNumber * (int)DataColumns.NumOfColumns + (int)dataColumn] - this[(rowNumber + 1) * (int)DataColumns.NumOfColumns + (int)dataColumn]) / this[(rowNumber + 1) * (int)DataColumns.NumOfColumns + (int)dataColumn];
        }

        public double GetLastChange(DateTime date, DataColumns dataColumn)
        {
            int rowNumber = GetDayNum(date);
            return (this[rowNumber * (int)DataColumns.NumOfColumns + (int)dataColumn] - this[(rowNumber + 1) * (int)DataColumns.NumOfColumns + (int)dataColumn]) / this[(rowNumber + 1) * (int)DataColumns.NumOfColumns + (int)dataColumn];
        }

        public double GetContinuousChange(int rowNumber, DataColumns dataColumn, int range)
        {
            double lastChange = GetLastChange(rowNumber, dataColumn);

            for (int i = rowNumber; i < rowNumber + range; i++)
            {
                double newLastChange = GetLastChange(i, dataColumn);
                if (lastChange > 0 && newLastChange < 0)
                {
                    return 0.0;
                }
                else if (lastChange < 0 && newLastChange > 0)
                {
                    return 0.0;
                }
                else
                {
                    lastChange += newLastChange;
                }
            }

            return lastChange;
        }

        public void LoadDataFromFile(string filePath)
        {
            string lastDataLine = string.Empty;
            using (StreamReader csvFile = new StreamReader(filePath))
            {
                // Read the first line and validate correctness of columns in the data file
                MapColumnNames(csvFile.ReadLine());

                while (!csvFile.EndOfStream)
                {
                    string dataLine = csvFile.ReadLine();

                    if (lastDataLine != dataLine)
                    {
                        Add(dataLine);
                    }

                    lastDataLine = dataLine;
                }
            }
        }

        public void SaveDataToFile(string filePath)
        {
            using (StreamWriter csvFile = new StreamWriter(filePath))
            {
                // Write the first line
                csvFile.WriteLine(GetColumnNamesString());

                for (int rowNum = 0; rowNum < NumOfRows; rowNum++)
                {                    
                    csvFile.WriteLine(GetDataString(rowNum));
                }
            }
        }

        private bool Add(string dataLine)
        {
            string[] data = dataLine.Split(',');
            if (data.Length < (int)DataColumns.NumOfColumns)
            {
                return false;
            }

            List<double> newDateData = new List<double>();


            foreach (int dataColumn in m_ColumnsMap.Keys.OrderBy(x => m_ColumnsMap[x]))
            {               
                if (string.IsNullOrWhiteSpace(data[dataColumn]))// || ((int)DataColumns.Volume == m_ColumnsMap[dataColumn] && Convert.ToDouble(data[dataColumn]) == 0.0))
                {
                    return false;
                }
                else
                {
                    if (m_ColumnsMap[dataColumn] == (int)DataColumns.Date)
                    {
                        newDateData.Add(Convert.ToDateTime(data[dataColumn]).Date.Ticks);
                    }
                    else
                    {
                        newDateData.Add(Convert.ToDouble(data[dataColumn]));
                    }
                }
            }

            AddRange(newDateData);

            return true;
        }

        public void AddTodayOpenData(DateTime date, double openPrice)
        {
            if (Contains(date.Ticks))
            {
                Console.WriteLine(" {0} Warning: No today trade data is available", DataSetCode);
                return;
            }

            List<double> todayData = new List<double>();
            for (int i = 0; i < (int)DataColumns.NumOfColumns; i++)
            {
                todayData.Add(0.0);
            }

            todayData[(int)DataColumns.Date] = date.Ticks;
            todayData[(int)DataColumns.Open] = openPrice;
            InsertRange(0, todayData);

            Console.WriteLine("{0} Contains new open data for {1}", DataSetCode, date.ToShortDateString());
            MapDates();
        }

        public override string ToString()
        {
            return DataSetCode;
        }

        internal void DeleteRows(DateTime date)
        {
            RemoveRange(0, GetDayNum(date) * (int)DataColumns.NumOfColumns);
            MapDates();
        }

        internal void CleanTodayData()
        {
            for (int i = 0; i < (int)DataColumns.NumOfColumns; i++)
            {
                if (i != (int)DataColumns.Date && i != (int)DataColumns.Open)
                {
                    this[i] = 0.0;
                }
            }
        }

        public bool ContainsTradeDay(DateTime day)
        {
            if (!m_DateMap.Keys.Contains(day))
            {
                return false;
            }

            if (m_DateMap.Keys.First() == day)
            {
                return true;
            }

            if (m_DateMap[day] == m_DateMap[day.AddDays(-1)])
            {
                return false;
            }

            return true;
        }

        public bool IsTradableDay(DateTime day)
        {
            if (!ContainsTradeDay(day))
            {
                return false;
            }

            int dayNum = GetDayNum(day);
            if (GetData(dayNum, DataColumns.Open) == GetData(dayNum, DataColumns.High)
                && GetData(dayNum, DataColumns.Open) == GetData(dayNum, DataColumns.Low)
                && GetData(dayNum, DataColumns.Open) == GetData(dayNum, DataColumns.Close))
            {
                return false;
            }

            return true;
        }

        #endregion

        #region Private Methods
        private void MapColumnNames(string columnNamesLine)
        {
            string[] columnNames = columnNamesLine.Split(',');

            for (int fileColumn = 0; fileColumn < columnNames.Length; fileColumn++)
            {
                if (m_ColumnNamesMap.Keys.Contains(columnNames[fileColumn].ToLower()))
                {
                    m_ColumnsMap.Add(fileColumn, m_ColumnNamesMap[columnNames[fileColumn].ToLower()]);
                }
            }

            if (m_ColumnsMap.Count < (int)DataColumns.NumOfColumns)
            {
                throw new Exception();
            }
        }

        private string GetDataString(int rowNum)
        {
            string dataLine = new DateTime((long)this[rowNum * (int)DataColumns.NumOfColumns]).ToShortDateString();
            for (int dataColumn = 1; dataColumn < (int)DataColumns.NumOfColumns; dataColumn++)
            {
                dataLine += "," + this[rowNum * (int)DataColumns.NumOfColumns + dataColumn];
            }

            return dataLine;
        }

        private string GetColumnNamesString()
        {
            string columnNames = ((DataColumns)0).ToString();

            for (int dataColumn = 1; dataColumn < (int)DataColumns.NumOfColumns; dataColumn++)
            {
                columnNames += "," + ((DataColumns)dataColumn).ToString();
            }

            return columnNames;
        }

        private void MapDates()
        {
            m_DateMap.Clear();
            DateTime lastDate = new DateTime((long)this[(int)DataColumns.Date]);
            DateTime firstDate = new DateTime((long)this[Count - (int)DataColumns.NumOfColumns + (int)DataColumns.Date]);

            int rownNum = NumOfRows - 1;
            for (DateTime dateTime = firstDate; dateTime <= lastDate; dateTime = dateTime.AddDays(1))
            {
                if (new DateTime((long)this[rownNum * (int)DataColumns.NumOfColumns + (int)DataColumns.Date]) != dateTime)
                {
                    m_DateMap.Add(dateTime, rownNum + 1);
                }
                else
                {
                    m_DateMap.Add(dateTime, rownNum);
                    rownNum--;
                }
            }
        }

        #endregion
    }
}
