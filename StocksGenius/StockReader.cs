using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public class StockReader
    {
        #region Members

        List<List<string>> m_FileData = null;

        #endregion

        #region Properties

        private Dictionary<string, List<float>> m_StockDataDelta = new Dictionary<string, List<float>>();

        public Dictionary<string, List<float>> StockDataDelta
        {
            get { return m_StockDataDelta; }
            set { m_StockDataDelta = value; }
        }

        private List<string> m_ColumnNames = new List<string>();

        public List<string> ColumnNames
        {
            get { return m_ColumnNames; }
            set { m_ColumnNames = value; }
        }

        private int m_DataSize = 0;

        public int DataSize
        {
            get { return m_DataSize; }
            set { m_DataSize = value; }
        }


        #endregion

        #region Constructors

        public StockReader(string filePath)
        {
            LoadCSVFile(filePath);
            LoadDeltaData();
        }

        #endregion

        #region Interface

        public void LoadCSVFile(string filePath)
        {
            m_FileData = new List<List<string>>();
            StreamReader csvFile = new StreamReader(filePath);
            
            while(!csvFile.EndOfStream)
            {
                string rawData = csvFile.ReadLine();
                m_FileData.Add(rawData.Split(',').ToList());
            }

            m_DataSize = m_FileData.Count - 1;
        }

        public void LoadDeltaData()
        {
            for (int i = 1; i < m_FileData[0].Count; i++)
            {
                string key = m_FileData[0][i];
                m_ColumnNames.Add(key);
                List<float> columnData = new List<float>();
                
                for (int j = 1; j < m_FileData.Count - 1; j++)
                {
                    float currentValue = (string.IsNullOrWhiteSpace(m_FileData[j][i])) ? 0 : (float)Convert.ChangeType(m_FileData[j][i], typeof(float));
                    float lastDayValue = (string.IsNullOrWhiteSpace(m_FileData[j + 1][i])) ? 0 : (float)Convert.ChangeType(m_FileData[j + 1][i], typeof(float));
                    columnData.Add((currentValue - lastDayValue) / lastDayValue);
                }

                m_StockDataDelta.Add(key, columnData);
            }
        }

        #endregion

    }
}
