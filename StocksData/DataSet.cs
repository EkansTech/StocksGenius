using System;
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

        static List<string> m_ColumnNames = new List<string>() { "Date", "Open", "High", "Low", "Close", "Volume" };

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

        public DataSet(string dataSetCode, string filePath, TestDataAction testDataAction = TestDataAction.None)
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
            }
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
            for (int i = 0; i < NumOfRows; i++)
            {
                if (GetDate(i) <= date)
                {
                    return i;
                }
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
                ValidateColumnNames(csvFile.ReadLine());

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

        private bool Add(string dataLine)
        {
            string[] data = dataLine.Split(',');

            List<double> newDateData = new List<double>();

            Add(Convert.ToDateTime(data[0]).Date.Ticks);
            
            for (int i = 1; i < (int)DataColumns.NumOfColumns; i++)
            {
                if (string.IsNullOrWhiteSpace(data[i]))
                {
                    return false;
                }
                else
                {
                    newDateData.Add(Convert.ToDouble(data[i]));
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
        }

        public override string ToString()
        {
            return DataSetCode;
        }

        internal void DeleteRows(DateTime date)
        {
            RemoveRange(0, GetDayNum(date) * (int)DataColumns.NumOfColumns);
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

        #endregion

        #region Private Methods
        private void ValidateColumnNames(string columnNamesLine)
        {
            string[] columnNames = columnNamesLine.Split(',');

            if (m_ColumnNames.Count != columnNames.Length)
            {
                //throw new Exception(string.Format("Not compatible columns in the {0} data set", DataSetName));
            }

            for (int i = 0; i < m_ColumnNames.Count; i++)
            {
                if (!m_ColumnNames[i].ToLower().Equals(columnNames[i].ToLower().Trim()))
                {
                    throw new Exception(string.Format("Expected column {0] instead of {1] in the {2} data set", m_ColumnNames[i], columnNames[i], DataSetCode));
                }
            }
        }

        #endregion
    }
}
