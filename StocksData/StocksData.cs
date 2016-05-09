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

        private List<string> m_DataSetsCodes;

        public List<string> DataSetsCodes
        {
            get { return m_DataSetsCodes; }
            set { m_DataSetsCodes = value; }
        }

        private Dictionary<string, string> m_DataSetPaths = new Dictionary<string, string>();

        public Dictionary<string, string> DataSetPaths
        {
            get { return m_DataSetPaths; }
            set { m_DataSetPaths = value; }
        }

        private Dictionary<string, string> m_PriceDataSetPaths = new Dictionary<string, string>();

        public Dictionary<string, string> PriceDataSetPaths
        {
            get { return m_PriceDataSetPaths; }
            set { m_PriceDataSetPaths = value; }
        }

        private Dictionary<string, string> m_DataPredictionsPaths = new Dictionary<string, string>();

        public Dictionary<string, string> DataPredictionsPaths
        {
            get { return m_DataPredictionsPaths; }
            set { m_DataPredictionsPaths = value; }
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
            get { if (m_DataPredictions.Count == 0) { LoadPredictions(DSSettings.MinimumRelevantPredictionResult); }  return m_DataPredictions; }
            set { m_DataPredictions= value; }
        }

        public StocksDataSource DataSource { get; set; }


        public string WorkingDirectory { get; set; }

        #endregion

        #region Constructor

        public StocksData(string workingDirectory, StocksDataSource dataSource = null)
        {
            DataSource = (dataSource == null) ? new QuandlDataSource() : dataSource;
            WorkingDirectory = workingDirectory;

            m_DataSetsCodes = DataSource.GetDataSetsList(workingDirectory);

            foreach (string dataSet in m_DataSetsCodes)
            {
                m_DataSetPaths.Add(dataSet, workingDirectory + DSSettings.DataSetsDir + dataSet + ".csv");
                m_PriceDataSetPaths.Add(dataSet, workingDirectory + DSSettings.PriceDataSetsDirectory + dataSet + ".csv");
                m_DataPredictionsPaths.Add(dataSet, workingDirectory + DSSettings.PredictionDir + dataSet + DSSettings.PredictionSuffix + ".csv");
            }
        }

        #endregion

        #region Interface

        public List<PredictionRecord> LoadPredictions(double effectivePredictionResult)
        {
            List<PredictionRecord> predictionRecords = new List<PredictionRecord>();
            foreach (string dataSetName in m_DataSetsCodes)
            {
                DataPredictions dataPredictions = new DataPredictions(m_DataSetPaths[dataSetName], m_DataPredictionsPaths[dataSetName]);
                if (!m_DataPredictions.ContainsKey(dataSetName))
                {
                    m_DataPredictions.Add(dataSetName, dataPredictions);
                }
                predictionRecords.AddRange(dataPredictions.GetBestPredictions(effectivePredictionResult));
            }

            return predictionRecords;
        }

        public void LoadDataPredictions()
        {
            foreach (string dataSetName in m_DataSetsCodes)
            {
                DataPredictions dataPredictions = new DataPredictions(m_DataSetPaths[dataSetName], m_DataPredictionsPaths[dataSetName]);
                if (!m_DataPredictions.ContainsKey(dataSetName))
                {
                    m_DataPredictions.Add(dataSetName, dataPredictions);
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
            foreach (string dataSetName in m_DataSetsCodes)
            {
                //if (datasetNumber > 0)
                //{
                //    Console.SetCursorPosition(0, Console.CursorTop - 2);
                //}

                dataSetNumber++;

                //Console.Write(new string(' ', Console.WindowWidth));
                //Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Current Stock: {0}", dataSetName);
                Console.WriteLine("Completed {0}%", (((double)dataSetNumber) / (double)m_DataSetsCodes.Count * 100.0).ToString("0.00"));


                DataSet dataSet = new DataSet(m_DataSetPaths[dataSetName]);
                DateTime timePoint = DateTime.Now;
                DataPredictions dataPredictions = new DataPredictions(dataSet, predictionsDirectory, true);
                loadTime += (double)(DateTime.Now - timePoint).TotalMilliseconds;
                gpuTime += dataPredictions.GPULoadTime;
                dataPredictions.SaveDataToFile(predictionsDirectory);
            }

            Console.WriteLine(string.Format("Prediction time = {0}, GPU total time - {1}", loadTime / 1000, gpuTime / 1000));
            Console.WriteLine();
            Console.ReadKey();

            return;
        }

        public void AnalyzeChangesEffects()
        {
            string latestpredictionsDirectory = WorkingDirectory + DSSettings.LatestPredictionsDir;
            string currentProject = WorkingDirectory.Split(Path.DirectorySeparatorChar).Last(x => !string.IsNullOrWhiteSpace(x));
            string latestPredictionsFilePath = latestpredictionsDirectory + currentProject + DSSettings.LatestPredictionsSuffix + ".csv";
            
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

                
                DataSet dataSet = new DataSet(dataSetsPath);
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
            foreach (string dataSetName in m_DataSetsCodes)
            {
                if (!m_DataSets.ContainsKey(dataSetName))
                {
                    m_DataSets.Add(dataSetName, new DataSet(m_DataSetPaths[dataSetName]));
                }
                if (!m_PriceDataSets.ContainsKey(dataSetName))
                {
                    m_PriceDataSets.Add(dataSetName, new DataSet(m_PriceDataSetPaths[dataSetName]));
                }
            }
        }

        public void ReloadDataSets()
        {
            if (m_DataPredictions.Count == 0)
            {
                LoadDataPredictions();
            }

            foreach (string dataSetName in m_DataSetsCodes)
            {
                if (!m_DataSets.ContainsKey(dataSetName))
                {
                    m_DataSets.Add(dataSetName, new DataSet(m_DataSetPaths[dataSetName]));
                }
                else
                {
                    m_DataSets[dataSetName] = new DataSet(m_DataSetPaths[dataSetName]);
                }
                if (!m_PriceDataSets.ContainsKey(dataSetName))
                {
                    m_PriceDataSets.Add(dataSetName, new DataSet(m_PriceDataSetPaths[dataSetName]));
                }
                else
                {
                    m_PriceDataSets[dataSetName] = new DataSet(m_PriceDataSetPaths[dataSetName]);
                }
                m_DataPredictions[dataSetName].DataSet = m_DataSets[dataSetName];
            }
        }

        public void MoveToDate(int dayNum)
        {
            foreach(string dataSetName in DataSets.Keys)
            {
                DataSets[dataSetName].DeleteRows(dayNum);
                DataSets[dataSetName].CleanTodayData();
                PriceDataSets[dataSetName].DeleteRows(dayNum);
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

        #endregion

        #region Private Methods

        #endregion
    }
}
