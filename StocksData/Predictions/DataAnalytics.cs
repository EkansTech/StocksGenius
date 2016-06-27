using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public struct Average
    {
        int Count;
        double AverageValue;

        public void Add(double value)
        {
            AverageValue = (AverageValue * Count + value) / (Count + 1);
            Count++;
        }
    }

    public class DefaultDictionary<Key, Value> : Dictionary<Key, Value>
    {
        public new Value this[Key key]
        {
            get
            {
                if (!ContainsKey(key))
                {
                    Add(key, (Value)typeof(Value).GetConstructor(null).Invoke(null));
                }

                return base[key];
            }
            set
            {
                if (!ContainsKey(key))
                {
                    Add(key, value);
                }
                else
                {
                    base[key] = value;
                }
            }
        }
    }

    public class DataAnalytics
    {
        #region Properties

        private string m_FileName = string.Empty;

        public string FileName
        {
            get { return m_FileName; }
            set { m_FileName = value; }
        }

        public DataSet DataSet { get; internal set; }

        public int LongestUpSequence { get; set; }

        public int LongestDownSequence { get; set; }

        public DefaultDictionary<int, Average> AverrageUpAfterDownSequence { get; set; }

        public DefaultDictionary<int, Average> AverrageDownAfterUpSequence { get; set; }

        public DefaultDictionary<int, Average> AverrageUpAfterDownPercentage { get; set; }

        public DefaultDictionary<int, Average> AverrageDownAfterUpPercentage { get; set; }

        public DefaultDictionary<int, int> NumberOfUpsAfterDownSequence { get; set; }

        public DefaultDictionary<int, int> NumberOfDownsAfterDownSequence { get; set; }

        public DefaultDictionary<int, int> NumberOfUpsAfterUpSequence { get; set; }

        public DefaultDictionary<int, int> NumberOfDownsAfterUpSequence { get; set; }

        #endregion

        #region Constructors

        protected DataAnalytics()
        {

        }

        private DataAnalytics(DataAnalytics DataAnalytics)
        {
            DataSet = DataAnalytics.DataSet;
            FileName = DataAnalytics.FileName;
        }

        public DataAnalytics(string dataPredictionFilePath)
        {
            LoadFromFile(dataPredictionFilePath);
            m_FileName = Path.GetFileNameWithoutExtension(dataPredictionFilePath);
            // CaclulatePredictions();
        }

        public DataAnalytics(string dataSetCode, string dataSetFilePath, string dataAnalyticsFilePath)
        {
            DataSet = new DataSet(dataSetCode, dataSetFilePath);
            LoadFromFile(dataAnalyticsFilePath);
            m_FileName = Path.GetFileNameWithoutExtension(dataAnalyticsFilePath);
            // CaclulatePredictions();
        }

        public DataAnalytics(string filePath, DataSet dataSet)
        {
            LoadFromFile(filePath);
            DataSet = dataSet;
            m_FileName = Path.GetFileNameWithoutExtension(filePath);
           // CaclulatePredictions();
        }

        public DataAnalytics(DataSet dataSet, string analyticsFilePath, bool useGPU = true)
        {
            DataSet = dataSet;
            m_FileName = Path.GetFileNameWithoutExtension(analyticsFilePath);
            string filePath = analyticsFilePath;

            if (File.Exists(filePath))
            {
                LoadFromFile(filePath, true);
            }

            LoadFromDataSet(useGPU);
        }

        #endregion

        #region Interface

        public void LoadFromDataSet(bool useGPU)
        {
            int upSequence = 0;
            int downSequence = 0;
            double upPercentage = 0.0;
            double downPercentage = 0.0;
            LongestUpSequence = 0;
            LongestDownSequence = 0;
            AverrageUpAfterDownSequence = new DefaultDictionary<int, Average>();
            AverrageDownAfterUpSequence = new DefaultDictionary<int, Average>();
            AverrageUpAfterDownPercentage = new DefaultDictionary<int, Average>();
            AverrageDownAfterUpPercentage = new DefaultDictionary<int, Average>();
            NumberOfUpsAfterDownSequence = new DefaultDictionary<int, int>();
            NumberOfDownsAfterDownSequence = new DefaultDictionary<int, int>();
            NumberOfUpsAfterUpSequence = new DefaultDictionary<int, int>();
            NumberOfDownsAfterUpSequence = new DefaultDictionary<int, int>();

            double lastOpenValue = DataSet.GetData(DataSet.NumOfRows - 1, DataSet.DataColumns.Open);

            for (int rowNum = DataSet.NumOfRows - 2; rowNum > 0; rowNum--)
            {
                if (!DataSet.IsTradableDay(DataSet.GetDate(rowNum)))
                {
                    continue;
                }

                double openValue = DataSet.GetData(DataSet.NumOfRows - 1, DataSet.DataColumns.Open);
                double change = (openValue - lastOpenValue) / lastOpenValue;

                if (change > 0 && upSequence > 0)
                {
                    NumberOfUpsAfterUpSequence[upSequence]++;
                    upSequence++;
                    upPercentage += (1 + upPercentage / 100.0) * change;
                }

                if (change > 0 && downSequence > 0)
                {
                    NumberOfUpsAfterDownSequence[downSequence]++;
                    AverrageUpAfterDownSequence[downSequence].Add(change);
                    AverrageUpAfterDownPercentage[(int)downPercentage].Add(change);
                    upSequence++;
                    downSequence = 0;
                    upPercentage += (1 + upPercentage / 100.0) * change;
                    downPercentage = 0.0;
                }

                if (change < 0 && upSequence > 0)
                {
                    NumberOfDownsAfterUpSequence[upSequence]++;
                    AverrageDownAfterUpSequence[upSequence].Add(change);
                    AverrageDownAfterUpPercentage[(int)upPercentage].Add(change);
                    downSequence++;
                    upSequence = 0;
                    downPercentage += (1 + downPercentage / 100.0) * change;
                    upPercentage = 0.0;
                }

                if (change < 0 && downSequence > 0)
                {
                    NumberOfDownsAfterDownSequence[downSequence]++;
                    downSequence++;
                    downPercentage += (1 + downPercentage / 100.0) * change;
                }
            }

    }

        public virtual void LoadFromFile(string filePath, bool loadBadPredictions = false)
        {
        }

        public virtual void SaveDataToFile(string folderPath)
        {
            using (StreamWriter writer = new StreamWriter(folderPath + "\\" + m_FileName))
            {

            }
        }

        public virtual void Add(string dataLine)
        {
        }

        #endregion

        #region Private Methods
        
        #endregion
    }
}
