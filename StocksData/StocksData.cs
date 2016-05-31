  using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class StocksData
    {
        #region Properties

        private DataSetsMetaData m_MetaData = null;

        public DataSetsMetaData MetaData
        {
            get { return m_MetaData; }
            set { m_MetaData = value; }
        }

        private Dictionary<string, DataSet> m_DataSets = new Dictionary<string, DataSet>();

        public Dictionary<string, DataSet> DataSets
        {
            get { return m_DataSets; }
            set { m_DataSets = value; }
        }

        private Dictionary<string, DataSet> m_PriceDataSets = new Dictionary<string, DataSet>();

        public Dictionary<string, DataSet> PriceDataSets
        {
            get { return m_PriceDataSets; }
            set { m_PriceDataSets = value; }
        }

        private Dictionary<string, DataPredictions> m_DataPredictions = new Dictionary<string, DataPredictions>();

        public Dictionary<string, DataPredictions> DataPredictions
        {
            get { if (m_DataPredictions.Count == 0) { LoadPredictions(DSSettings.EffectivePredictionResult); }  return m_DataPredictions; }
            set { m_DataPredictions= value; }
        }

        public StocksDataSource DataSource { get; set; }


        public string WorkingDirectory { get; set; }

        private bool m_UseSimPredictions;

        public bool UseSimPredictions
        {
            get { return m_UseSimPredictions; }
            set { m_UseSimPredictions = value; }
        }


        #endregion

        #region Constructor

        public StocksData(string workingDirectory, DataSourceTypes dataSourceType = DataSourceTypes.Quandl, bool useSimPredictions = false)
        {
            m_UseSimPredictions = useSimPredictions;
            DataSource = NewDataSource(dataSourceType);
            WorkingDirectory = workingDirectory;
            m_MetaData = new DataSetsMetaData(WorkingDirectory);
        }

        #endregion

        #region Interface

        public List<PredictionRecord> LoadPredictions(double effectivePredictionResult)
        {
            List<PredictionRecord> predictionRecords = new List<PredictionRecord>();
            foreach (string dataSetCode in MetaData.Keys)
            {
                DataPredictions dataPredictions;
                if (UseSimPredictions)
                {
                    dataPredictions = new DataPredictions(dataSetCode, MetaData[dataSetCode].DataSetFilePath, MetaData[dataSetCode].SimDataPredictionsFilePath);

                }
                else
                {
                    dataPredictions = new DataPredictions(dataSetCode, MetaData[dataSetCode].DataSetFilePath, MetaData[dataSetCode].DataPredictionsFilePath);
                }
                if (!m_DataPredictions.ContainsKey(dataSetCode))
                {
                    m_DataPredictions.Add(dataSetCode, dataPredictions);
                }
                predictionRecords.AddRange(dataPredictions.GetBestPredictions(effectivePredictionResult));
            }

            return predictionRecords;
        }

        public void LoadDataPredictions()
        {
            foreach (string dataSetCode in MetaData.Keys)
            {
                DataPredictions dataPredictions;
                if (UseSimPredictions)
                {
                    dataPredictions = new DataPredictions(dataSetCode, MetaData[dataSetCode].DataSetFilePath, MetaData[dataSetCode].SimDataPredictionsFilePath);

                }
                else
                {
                    dataPredictions = new DataPredictions(dataSetCode, MetaData[dataSetCode].DataSetFilePath, MetaData[dataSetCode].DataPredictionsFilePath);
                }
                if (!m_DataPredictions.ContainsKey(dataSetCode))
                {
                    m_DataPredictions.Add(dataSetCode, dataPredictions);
                }
            }
        }


        public void BuildDataPredictions()
        {
            string predictionsDirectory = WorkingDirectory + DSSettings.PredictionDir;
            if (!Directory.Exists(predictionsDirectory))
            {
                Directory.CreateDirectory(predictionsDirectory);
            }

            int dataSetNumber = 0;

            double loadTime = 0;
            double gpuTime = 0;
            foreach (string dataSetCode in MetaData.Keys)
            {
                Console.WriteLine("Current Stock: {0}", dataSetCode);
                Console.WriteLine("Completed {0}%", (((double)dataSetNumber) / (double)MetaData.Count * 100.0).ToString("0.00"));


                DataSet dataSet = new DataSet(dataSetCode, MetaData[dataSetCode].DataSetFilePath);
                DateTime timePoint = DateTime.Now;
                DataPredictions dataPredictions = new DataPredictions(dataSet, MetaData[dataSetCode].DataPredictionsFilePath, true);
                loadTime += (double)(DateTime.Now - timePoint).TotalMilliseconds;
                gpuTime += dataPredictions.GPULoadTime;
                dataPredictions.SaveDataToFile(predictionsDirectory);

                dataSetNumber++;
            }

            Console.WriteLine(string.Format("Prediction time = {0}, GPU total time - {1}", loadTime / 1000, gpuTime / 1000));
            Console.WriteLine();
            Console.ReadKey();

            return;
        }


        public void BuildSimDataPredictions(DateTime startDate, int jump, TimeType predictionTimeType, DateTime endDate, TestDataAction testDataType = TestDataAction.LoadDataUpTo, int relevantMonths = 60, TimeType dataTimeType = TimeType.Month, string suffix = null)
        {
            string rootPredictionsDirectory = MetaData.SimPredictionDir + (suffix != null ? suffix + "\\" : string.Empty);
            if (!Directory.Exists(rootPredictionsDirectory))
            {
                Directory.CreateDirectory(rootPredictionsDirectory);
            }
            Dictionary<string, DataSet> dataSets = new Dictionary<string, DataSet>();
            foreach (string dataSetCode in MetaData.Keys)
            {
                dataSets.Add(dataSetCode, new DataSet(dataSetCode, MetaData[dataSetCode].DataSetFilePath));
            }

            while (startDate < endDate)
            {
                string predictionsDirectory = rootPredictionsDirectory + startDate.ToShortDateString().Replace("/","_") + "\\";
                if (!Directory.Exists(predictionsDirectory))
                {
                    Directory.CreateDirectory(predictionsDirectory);
                }

                Console.WriteLine("Predictions for {0}", startDate.ToShortDateString());
                int dataSetNumber = 0;

                double loadTime = 0;
                double gpuTime = 0;
                foreach (string dataSetCode in MetaData.Keys)
                {
                    Console.WriteLine("Current Stock: {0}", dataSetCode);
                    Console.WriteLine("Completed {0}%", (((double)dataSetNumber) / (double)MetaData.Count * 100.0).ToString("0.00"));
                    MetaData[dataSetCode].SimPredictionsDir = predictionsDirectory;

                    DataSet dataSet = new DataSet(dataSets[dataSetCode], testDataType, startDate, relevantMonths);
                    DateTime timePoint = DateTime.Now;
                    DataPredictions dataPredictions = new DataPredictions(dataSet, MetaData[dataSetCode].SimDataPredictionsFilePath, true);
                    loadTime += (double)(DateTime.Now - timePoint).TotalMilliseconds;
                    gpuTime += dataPredictions.GPULoadTime;
                    dataPredictions.SaveDataToFile(predictionsDirectory);

                    dataSetNumber++;
                }

                Console.WriteLine(string.Format("Prediction time = {0}, GPU total time - {1}", loadTime / 1000, gpuTime / 1000));
                Console.WriteLine();

                switch (predictionTimeType)
                {
                    case TimeType.Day:
                        startDate = startDate.AddDays(jump);
                        break;
                    case TimeType.Week:
                        startDate = startDate.AddDays(jump * 7);
                        break;
                    case TimeType.Month:
                        startDate = startDate.AddMonths(jump);
                        break;
                    case TimeType.Year:
                        startDate = startDate.AddYears(jump);
                        break;
                }
            }
            Console.ReadKey();
        }


        public void BuildCombinedDataPredictions()
        {
            ReloadDataSets();

            List<DataPredictions> dataPredictionsList = new List<DataPredictions>();
            foreach (string dataSetCode in m_MetaData.Keys)
            {
                dataPredictionsList.Add(new DataPredictions(m_MetaData[dataSetCode].SimDataPredictionsFilePath));
            }

            CombinedDataPredictions combinedDataPredictions = new CombinedDataPredictions(dataPredictionsList);

            combinedDataPredictions.SaveDataToFile(m_MetaData.SimCombinedDataPredictionsFilePath);

            return;
        }


        public void BuildSimDataPredictions()
        {
            string predictionsDirectory = WorkingDirectory + DSSettings.SimPredictionDir;
            if (!Directory.Exists(predictionsDirectory))
            {
                Directory.CreateDirectory(predictionsDirectory);
            }

            int dataSetNumber = 0;

            double loadTime = 0;
            double gpuTime = 0;
            foreach (string dataSetCode in MetaData.Keys)
            {
                Console.WriteLine("Current Stock: {0}", dataSetCode);
                Console.WriteLine("Completed {0}%", (((double)dataSetNumber) / (double)MetaData.Count * 100.0).ToString("0.00"));


                DataSet dataSet = new DataSet(dataSetCode, MetaData[dataSetCode].DataSetFilePath, TestDataAction.LoadWithoutTestData);
                DateTime timePoint = DateTime.Now;
                DataPredictions dataPredictions = new DataPredictions(dataSet, MetaData[dataSetCode].SimDataPredictionsFilePath, true);
                loadTime += (double)(DateTime.Now - timePoint).TotalMilliseconds;
                gpuTime += dataPredictions.GPULoadTime;
                dataPredictions.SaveDataToFile(predictionsDirectory);

                dataSetNumber++;
            }

            Console.WriteLine(string.Format("Prediction time = {0}, GPU total time - {1}", loadTime / 1000, gpuTime / 1000));
            Console.WriteLine();
            Console.ReadKey();

            return;
        }

        public void BuildiForexPredictions()
        {
            int datasetNumber = 0;
            string iForexPredictionsFolder = "\\iForexPrediction\\";
            string iForexStocksListFile = "iForexStocks.txt";

            Dictionary<string /*stock name*/, string /*stock dataset file*/> iForexFiles = LoadStocksListFile(WorkingDirectory + iForexStocksListFile);
            double loadTime = 0;
            double gpuTime = 0;
            foreach (string stockName in iForexFiles.Keys)
            {
                string dataSetsPath = WorkingDirectory + DSSettings.DataSetsDir + iForexFiles[stockName];
                //if (datasetNumber > 0)
                //{
                //    Console.SetCursorPosition(0, Console.CursorTop - 2);
                //}

                datasetNumber++;

                //Console.Write(new string(' ', Console.WindowWidth));
                //Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Current Stock: {0}", stockName);
                Console.WriteLine("Completed {0}%", (((double)datasetNumber) / (double)iForexFiles.Count * 100.0).ToString("0.00"));

                
                DataSet dataSet = new DataSet(stockName, dataSetsPath);
                DateTime timePoint = DateTime.Now;
                DataPredictions dataPredictions = new DataPredictions(dataSet, WorkingDirectory + iForexPredictionsFolder,true);
                loadTime += (double)(DateTime.Now - timePoint).TotalMilliseconds;
                gpuTime += dataPredictions.GPULoadTime;
                dataPredictions.SaveDataToFile(WorkingDirectory + iForexPredictionsFolder);
            }

            Console.WriteLine(string.Format("Prediction time = {0}, GPU total time - {1}", loadTime / 1000, gpuTime / 1000));
            Console.WriteLine();
            Console.ReadKey();

            return;
        }

        public static Dictionary<string, string> LoadStocksListFile(string filePath)
        {
            Dictionary<string, string> stocksList = new Dictionary<string, string>();
            using (StreamReader reader = new StreamReader(filePath))
            {
                while (!reader.EndOfStream)
                {
                    string line = reader.ReadLine();
                    stocksList.Add(line.Split(':')[0].Trim(), line.Split(':')[1].Trim());
                }
            }

            return stocksList;
        }

        public void LoadDataSets()
        {
            foreach (string dataSetCode in MetaData.Keys)
            {
                if (!m_DataSets.ContainsKey(dataSetCode))
                {
                    m_DataSets.Add(dataSetCode, new DataSet(dataSetCode, MetaData[dataSetCode].DataSetFilePath));
                }
                if (!m_PriceDataSets.ContainsKey(dataSetCode))
                {
                    m_PriceDataSets.Add(dataSetCode, new DataSet(dataSetCode, MetaData[dataSetCode].DataSetFilePath));
                }
            }
        }

        public void ReloadDataSets()
        {
            if (m_DataPredictions.Count == 0)
            {
                LoadDataPredictions();
            }

            foreach (string dataSetCode in MetaData.Keys)
            {
                if (!m_DataSets.ContainsKey(dataSetCode))
                {
                    m_DataSets.Add(dataSetCode, new DataSet(dataSetCode, MetaData[dataSetCode].DataSetFilePath));
                }
                else
                {
                    m_DataSets[dataSetCode] = new DataSet(dataSetCode, MetaData[dataSetCode].DataSetFilePath);
                }
                if (!m_PriceDataSets.ContainsKey(dataSetCode))
                {
                    m_PriceDataSets.Add(dataSetCode, new DataSet(dataSetCode, MetaData[dataSetCode].PriceDataSetFilePath));
                }
                else
                {
                    m_PriceDataSets[dataSetCode] = new DataSet(dataSetCode, MetaData[dataSetCode].PriceDataSetFilePath);
                }
                m_DataPredictions[dataSetCode].DataSet = m_DataSets[dataSetCode];
            }
        }

        public void MoveToDate(DateTime date)
        { 
            foreach(string dataSetName in DataSets.Keys)
            {
                DataSets[dataSetName].DeleteRows(date);
                DataSets[dataSetName].CleanTodayData();
            }

            foreach (string dataSetName in DataSets.Keys)
            {
                PriceDataSets[dataSetName].DeleteRows(date);
                PriceDataSets[dataSetName].CleanTodayData();
            }
        }

        public bool AreDataSetsSynchronized()
        {
            DateTime date = DataSets.Values.First().GetDate(0);

            foreach (string dataSetName in DataSets.Keys)
            {
               if (!DataSets[dataSetName].GetDate(0).Equals(date) || !PriceDataSets[dataSetName].GetDate(0).Equals(date))
                {
                    return false;
                }
            }

            return true;
        }

        public void AddOpenDataToDataSets(string openDataFile, string dataSetCodesPrefix)
        {
            Dictionary<string, iForexDataSource.OpenData> openData = iForexDataSource.LoadTodayOpenData(openDataFile, MetaData);

            foreach (string dataSetCode in openData.Keys)
            {
                m_DataSets[dataSetCode].AddTodayOpenData(openData[dataSetCode].Date, openData[dataSetCode].OpenValue);
                PriceDataSets[dataSetCode].AddTodayOpenData(openData[dataSetCode].Date, openData[dataSetCode].OpenValue);
            }
            //using (StreamReader reader = new StreamReader(openDataFile))
            //{
            //    while (!reader.EndOfStream)
            //    {
            //        string[] lineData = reader.ReadLine().Split(',');
            //        string dataSetName = dataSetCodesPrefix + lineData[0].Trim('"');
            //        if (lineData[1].Equals("N/A"))
            //        {
            //            Console.WriteLine("The open data for {0} is not available", dataSetName);
            //            return;
            //        }
            //        double openPrice = Convert.ToDouble(lineData[1]);
            //        string[] dateValues = lineData[2].Trim('"').Split('/');
            //        DateTime date = new DateTime(Convert.ToInt32(dateValues[2]), Convert.ToInt32(dateValues[0]), Convert.ToInt32(dateValues[1]));
            //        m_DataSets[dataSetName].AddTodayOpenData(date, openPrice);
            //    }
            //}
        }

        #endregion

        #region Private Methods

        private StocksDataSource NewDataSource(DataSourceTypes dataSourceType)
        {
            switch (dataSourceType)
            {
                case DataSourceTypes.Quandl: return new QuandlDataSource();
                case DataSourceTypes.Yahoo: return new YahooDataSource();
                case DataSourceTypes.Xignite: return new XigniteDataSource();
                case DataSourceTypes.Bloomberg: return new BloombergDataSource();
                default:
                    return new YahooDataSource();
            }
        }

        #endregion
    }
}
