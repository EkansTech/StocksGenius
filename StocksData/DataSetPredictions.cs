using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class PredictionsDataSet : List<double>
    {
        #region Enums

        public enum DataColumns
        {
            Depth,
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

        private string m_PredictionDataSetName = string.Empty;

        public string PredictionDataSetName
        {
            get { return m_PredictionDataSetName; }
            set { m_PredictionDataSetName = value; }
        }

        #endregion

        #region Constructors

        public PredictionsDataSet()
        {
        }

        public PredictionsDataSet(string filePath)
        {
            LoadFromFile(filePath);
            m_PredictionDataSetName = Path.GetFileNameWithoutExtension(filePath);
        }

        public PredictionsDataSet(ChangesDataSet changesDataSet, bool useGPU = true)
        {
            LoadFromChangesDataSet(changesDataSet, useGPU);
        }

        #endregion

        #region Interface

        public void LoadFromChangesDataSet(ChangesDataSet ChangesDataSet, bool useGPU = true)
        {
            m_PredictionDataSetName = ChangesDataSet.ChangeDataSetName.Substring(0, ChangesDataSet.ChangeDataSetName.IndexOf(Constants.ChangeDataSetSuffix)) + Constants.PredictionDataSetSuffix;
            
            if (useGPU)
            {
                CalculatePredictionsForDataTypeGPU(ChangesDataSet);
            }
            else
            {
                CalculatePredictionsForDataTypeCPU(ChangesDataSet);
            }
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
            using (StreamWriter csvFile = new StreamWriter(folderPath + "\\" + PredictionDataSetName + ".csv"))
            {
                // Write the first line 
                csvFile.WriteLine(GetColumnNamesString());
                for (int currentDate = 0; currentDate < NumOfRows; currentDate++)
                {
                    csvFile.WriteLine(GetDataString(currentDate));
                }
            }
        }

        public void Add(string dataLine)
        {
            string[] data = dataLine.Split(',');

            Add(Convert.ToDouble(data[0]));
            Add(Convert.ToDateTime(data[1]).Ticks);

            for (int i = 2; i < (int)DataColumns.NumOfColumns; i++)
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

        private void CalculatePredictionsForDataTypeCPU(ChangesDataSet changesDataSet)
        {
            for (int depth = Constants.MinDepthRange; depth <= Constants.MaxDepthRange; depth++)
            {
                for (int tradeDate = 0; tradeDate < changesDataSet.NumOfRows - depth - 1; tradeDate++)
                {
                    Add(depth);
                    Add(changesDataSet.GetValue(tradeDate, ChangesDataSet.DataColumns.Date));

                    for (int j = 2; j < (int)DataColumns.NumOfColumns; j++)
                     {
                        double sum = 0;

                        for (int calcDate = tradeDate + 1; calcDate <= tradeDate + depth; calcDate++)
                        {
                            sum += changesDataSet.GetValue(calcDate, j);
                        }

                        Add(sum);
                    }
                }
            }
        }

        private void CalculatePredictionsForDataTypeGPU(ChangesDataSet changesDataSet)
        {
            AddRange(GPUPredictions.CalculatePredictions((changesDataSet as List<double>).ToArray(), changesDataSet.NumOfRows));
        }

        private string GetDataString(int currentRow)
        {
            string dataString = this[currentRow * (int)DataColumns.NumOfColumns].ToString();

            dataString += "," + (new DateTime((long)this[currentRow * (int)DataColumns.NumOfColumns + 1])).ToString();

            for (int i = 2; i < (int)DataColumns.NumOfColumns; i++)
            {
                dataString += "," + this[currentRow * (int)DataColumns.NumOfColumns + i].ToString(); ;
            }

            return dataString;
        }

        private string GetColumnNamesString()
        {
            string columnNames = DataColumns.Depth.ToString();

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
                throw new Exception(string.Format("Not compatible columns in the {0} predictions data set", PredictionDataSetName));
            }

            for (int i = 0; i < (int)DataColumns.NumOfColumns; i++)
            {
                if (!((DataColumns)i).ToString().ToLower().Equals(columnNames[i].ToLower().Trim()))
                {
                    throw new Exception(string.Format("Expected column {0] instead of {1] in the {2} predictions data set", ((DataColumns)i).ToString(), columnNames[i], PredictionDataSetName));
                }
            }
        }

        #endregion
    }
}
