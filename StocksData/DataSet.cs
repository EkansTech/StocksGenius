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


        private string m_DataSetName = string.Empty;

        public string DataSetName
        {
            get { return m_DataSetName; }
            set { m_DataSetName = value; }
        }
        
        #endregion

        #region Constructor

        public DataSet()
        {
        }

        public DataSet(string filePath, TestDataAction testDataAction = TestDataAction.None)
        {
            LoadDataFromFile(filePath);
            m_DataSetName = Path.GetFileNameWithoutExtension(filePath);

            switch (testDataAction)
            {
                case TestDataAction.None:
                    break;
                case TestDataAction.LoadOnlyPredictionData:
                    RemoveRange(0, DSSettings.TestRange * (int)DataColumns.NumOfColumns);
                    RemoveRange(DSSettings.DataSetForPredictionsSize * (int)DataColumns.NumOfColumns, Count - DSSettings.DataSetForPredictionsSize * (int)DataColumns.NumOfColumns);
                    break;
                case TestDataAction.LoadOnlyTestData:
                    RemoveRange(DSSettings.TestMinSize * (int)DataColumns.NumOfColumns, Count - DSSettings.TestMinSize * (int)DataColumns.NumOfColumns);
                    break;
            }
        }

        #endregion

        #region Interface
        public List<double> GetDayData(int rowNumber)
        {
            return GetRange(rowNumber * (int)DataColumns.NumOfColumns, (int)DataColumns.NumOfColumns);
        }

        public void LoadDataFromFile(string filePath)
        {
            using (StreamReader csvFile = new StreamReader(filePath))
            {
                // Read the first line and validate correctness of columns in the data file
                ValidateColumnNames(csvFile.ReadLine());

                while (!csvFile.EndOfStream)
                {
                    Add(csvFile.ReadLine());
                }
            }
        }

        private void Add(string dataLine)
        {
            string[] data = dataLine.Split(',');

            Add(Convert.ToDateTime(data[0]).Ticks);
            
            for (int i = 1; i < (int)DataColumns.NumOfColumns; i++)
            {
                if (string.IsNullOrWhiteSpace(data[i]))
                {
                    Add(0);
                }
                else
                {
                    Add(Convert.ToDouble(data[i]));
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
                    throw new Exception(string.Format("Expected column {0] instead of {1] in the {2} data set", m_ColumnNames[i], columnNames[i], DataSetName));
                }
            }
        }

        #endregion
    }
}
