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

        private List<string> m_DataSets;

        public List<string> DataSets
        {
            get { return m_DataSets; }
            set { m_DataSets = value; }
        }

        private Dictionary<string, string> m_DataSetPaths = new Dictionary<string, string>();

        public Dictionary<string, string> DataSetPaths
        {
            get { return m_DataSetPaths; }
            set { m_DataSetPaths = value; }
        }

        private Dictionary<string, string> m_DataPredictionsPaths = new Dictionary<string, string>();

        public Dictionary<string, string> DataPredictionsPaths
        {
            get { return m_DataPredictionsPaths; }
            set { m_DataPredictionsPaths = value; }
        }

        private Dictionary<string, DataPredictions> m_DataPredictions = new Dictionary<string, DataPredictions>();

        public Dictionary<string, DataPredictions> DataPredictions
        {
            get { return m_DataPredictions; }
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

            m_DataSets = DataSource.GetDataSetsList(workingDirectory);

            foreach (string dataSet in m_DataSets)
            {
                m_DataSetPaths.Add(dataSet, workingDirectory + DSSettings.DataSetsDir + dataSet + ".csv");
                m_DataPredictionsPaths.Add(dataSet, workingDirectory + DSSettings.PredictionDir + dataSet + DSSettings.PredictionSuffix + ".csv");
            }
        }

        #endregion

        #region Interface

        public List<PredictionRecord> LoadPredictions(double effectivePredictionResult)
        {
            List<PredictionRecord> predictionRecords = new List<PredictionRecord>();
            foreach (string dataSetName in m_DataSets)
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
            foreach (string dataSetName in m_DataSets)
            {
                //if (datasetNumber > 0)
                //{
                //    Console.SetCursorPosition(0, Console.CursorTop - 2);
                //}

                dataSetNumber++;

                //Console.Write(new string(' ', Console.WindowWidth));
                //Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Current Stock: {0}", dataSetName);
                Console.WriteLine("Completed {0}%", (((double)dataSetNumber) / (double)m_DataSets.Count * 100.0).ToString("0.00"));


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

        #endregion

        #region Private Methods

        #endregion
    }
}
