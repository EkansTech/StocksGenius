using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class ChangesDataSet : List<double>
    {
        #region Enums

        public enum DataColumns
        {
            Date,
            OpenChange,
            HighChange,
            LowChange,
            CloseChange,
            VolumeChange,
            HighLowDif,
            HighOpenDif,
            LowOpenDif,
            CloseOpenDif,
            HighPrevCloseDif,
            LowPrevCloseDif,
            OpenPrevCloseDif,

            NumOfColumns,
        }

        #endregion

        #region Properties

        public int NumOfRows
        {
            get { return Count / (int)DataColumns.NumOfColumns; }
        }

        private string m_ChangeDataSetName = string.Empty;

        public string ChangeDataSetName
        {
            get { return m_ChangeDataSetName; }
            set { m_ChangeDataSetName = value; }
        }

        #endregion

        #region Constructors

        public ChangesDataSet()
        {
        }

        public ChangesDataSet(string filePath)
        {
            LoadFromFile(filePath);
            m_ChangeDataSetName = Path.GetFileNameWithoutExtension(filePath);
        }

        public ChangesDataSet(DataSet dataSet, bool useGPU = true)
        {
            if (useGPU)
            {
                LoadFromDataSetGPU(dataSet);
            }
            else
            {
                LoadFromDataSetCPU(dataSet);
            }
        }

        #endregion

        #region Interface

        public void LoadFromDataSetCPU(DataSet dataSet)
        {
            m_ChangeDataSetName = dataSet.DataSetName + Constants.ChangeDataSetSuffix;

            for (int offset = 0; offset < dataSet.Count - (int)DataSet.DataColumns.NumOfColumns * 2; offset += (int)DataSet.DataColumns.NumOfColumns)
            {
                Add(dataSet[offset]);
                CalculateChangesPercent(dataSet, offset, DataSet.DataColumns.Open);
                CalculateChangesPercent(dataSet, offset, DataSet.DataColumns.High);
                CalculateChangesPercent(dataSet, offset, DataSet.DataColumns.Low);
                CalculateChangesPercent(dataSet, offset, DataSet.DataColumns.Close);
                CalculateChangesPercent(dataSet, offset, DataSet.DataColumns.Volume);
                CalculateDifs(dataSet, offset, DataSet.DataColumns.High, DataSet.DataColumns.Low);
                CalculateDifs(dataSet, offset, DataSet.DataColumns.High, DataSet.DataColumns.Open);
                CalculateDifs(dataSet, offset, DataSet.DataColumns.Low, DataSet.DataColumns.Open);
                CalculateDifs(dataSet, offset, DataSet.DataColumns.Close, DataSet.DataColumns.Open);
                CalculateDifs(dataSet, offset, DataSet.DataColumns.High, DataSet.DataColumns.Close, true);
                CalculateDifs(dataSet, offset, DataSet.DataColumns.Low, DataSet.DataColumns.Close, true);
                CalculateDifs(dataSet, offset, DataSet.DataColumns.Open, DataSet.DataColumns.Close, true);
            }
        }

        public void LoadFromDataSetGPU(DataSet dataSet)
        {
            m_ChangeDataSetName = dataSet.DataSetName + Constants.ChangeDataSetSuffix;
            
            AddRange(GPUChanges.CalculateChangesInPercents((dataSet as List<double>).ToArray(), dataSet.NumOfRows));
        }

        private void CalculateChangesPercent(DataSet dataSet, int offset, DataSet.DataColumns sourceDataType)
        {
            double currentValue = dataSet[offset + (int)sourceDataType];
            double prevValue = dataSet[offset + (int)DataColumns.NumOfColumns + (int)sourceDataType];
            Add((currentValue - prevValue) / prevValue);
        }

        private void CalculateDifs(DataSet dataSet, int offset, DataSet.DataColumns difFromDataType, DataSet.DataColumns difDataType, bool usePrevDay = false)
        {
            double difFromValue = dataSet[offset + (int)difFromDataType];
            double difValue;
            if (usePrevDay)
            {
                difValue = dataSet[offset + (int)difDataType + (int)DataColumns.NumOfColumns];
            }
            else
            {
                difValue = dataSet[offset + (int)difDataType];
            }
            
            Add((difFromValue - difValue) / difValue);
        }

        public void LoadFromFile(string filePath)
        {
            StreamReader csvFile = new StreamReader(filePath);

            // Read the first line and validate correctness of columns in the data file
            ValidateColumnNames(csvFile.ReadLine());

            while (!csvFile.EndOfStream)
            {
                Add(csvFile.ReadLine());
            }
        }
        public void SaveDataToFile(string folderPath)
        {
            using (StreamWriter csvFile = new StreamWriter(folderPath + "\\" + ChangeDataSetName + ".csv"))
            {

                // Write the first line and validate correctness of columns in the data file
                csvFile.WriteLine(GetColumnNamesString());

                for (int i = 0; i < NumOfRows; i++)
                {
                    csvFile.WriteLine(GetDataString(i));
                }
            }
        }

        public void Add(string dataLine)
        {
            string[] data = dataLine.Split(',');

            Add(Convert.ToDateTime(data[0]).Ticks);

            for (int i = 1; i < (int)DataColumns.NumOfColumns; i++)
            {
                Add(Convert.ToDouble(data[i]));
            }
        }

        public double GetValue(int rowNumber, DataColumns column)
        {
            return this[rowNumber * (int)DataColumns.NumOfColumns + (int)column];
        }

        public double GetValue(int rowNumber, int column)
        {
            return this[rowNumber * (int)DataColumns.NumOfColumns + column];
        }

        #endregion

        #region Private Methods

        private string GetDataString(int curentDate)
        {
            string dataString =  (new DateTime((long)this[curentDate * (int)DataColumns.NumOfColumns])).ToString();

            for (int i = 1; i < (int)DataColumns.NumOfColumns; i++)
            {
                dataString += "," + this[curentDate * (int)DataColumns.NumOfColumns + i].ToString();
            }

            return dataString;
        }

        private string GetColumnNamesString()
        {
            string columnNames = DataColumns.Date.ToString();
            for (int i = 1; i < (int)DataColumns.NumOfColumns; i++)
            {
                columnNames += "," + ((DataColumns)i).ToString();
            }

            return columnNames;
        }

        private void ValidateColumnNames(string columnNamesLine)
        {
            string[] columnNames = columnNamesLine.Split(',');

            if (((int)DataColumns.NumOfColumns + 1) != columnNames.Length)
            {
                throw new Exception(string.Format("Not compatible columns in the {0} changes data set", ChangeDataSetName));
            }

            for (int i = 0; i < (int)DataColumns.NumOfColumns; i++)
            {
                if (!((DataColumns)i).ToString().ToLower().Equals(columnNames[i].ToLower().Trim()))
                {
                    throw new Exception(string.Format("Expected column {0] instead of {1] in the {2} changes data set", ((DataColumns)i).ToString(), columnNames[i], ChangeDataSetName));
                }
            }
        }

        #endregion
    }
}
