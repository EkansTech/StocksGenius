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

        private List<string> m_DataSetPaths;

        public List<string> DataSetPaths
        {
            get { return m_DataSetPaths; }
            set { m_DataSetPaths = value; }
        }

        private string m_StocksDataPath;

        public string StockDataPath
        {
            get { return m_StocksDataPath; }
            set { m_StocksDataPath = value; }
        }

        #endregion

        #region Constructor

        public StocksData(string path)
        {
            m_StocksDataPath = path;
            m_DataSetPaths = Directory.GetFiles(m_StocksDataPath + DSSettings.DataSetsDir).ToList();
        }

        #endregion

        #region Interface

        public void BuildiForexAnalyzer()
        {
            int datasetNumber = 0;
            string iForexAnalyzerFolder = "\\iForexAnalyzer\\";
            string iForexStocksListFile = "iForexStocks.txt";

            Dictionary<string /*stock name*/, string /*stock dataset file*/> iForexFiles = LoadStocksListFile(StockDataPath + iForexStocksListFile);
            float loadTime = 0;
            float gpuTime = 0;
            foreach (string stockName in iForexFiles.Keys)
            {
                string dataSetsPath = StockDataPath + DSSettings.DataSetsDir + iForexFiles[stockName];
                //if (datasetNumber > 0)
                //{
                //    Console.SetCursorPosition(0, Console.CursorTop - 2);
                //}

                datasetNumber++;

                //Console.Write(new string(' ', Console.WindowWidth));
                //Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Current Stock: {0}", stockName);
                Console.WriteLine("Completed {0}%", (((float)datasetNumber) / (float)iForexFiles.Count * 100.0).ToString("0.00"));

                
                DataSet dataSet = new DataSet(dataSetsPath, TestDataAction.RemoveTestData);
                DateTime timePoint = DateTime.Now;
                DataAnalyzer dataAnalyzer = new DataAnalyzer(dataSet, StockDataPath + iForexAnalyzerFolder,true);
                loadTime += (float)(DateTime.Now - timePoint).TotalMilliseconds;
                gpuTime += dataAnalyzer.GPULoadTime;
                dataAnalyzer.SaveDataToFile(StockDataPath + iForexAnalyzerFolder);
            }

            Console.WriteLine(string.Format("Analyzer time = {0}, GPU total time - {1}", loadTime / 1000, gpuTime / 1000));
            Console.WriteLine();
            Console.ReadKey();

            return;
        }

        public void BuildDataAnalyzers()
        {
            int datasetNumber = 0;

            Console.WriteLine("DataSet Analyzer");
            float loadTime = 0;
            foreach (string dataSetPath in m_DataSetPaths.Where(x => Path.GetFileName(x).StartsWith("WIKI-AA.csv")))
            {
                if (datasetNumber > 0)
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 2);
                }

                datasetNumber++;

                Console.Write(new string(' ', Console.WindowWidth));
                Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Current Data Set: {0}", Path.GetFileName(dataSetPath));
                Console.WriteLine("Completed {0}%", (((float)datasetNumber) / (float)m_DataSetPaths.Count * 100.0).ToString("0.00"));


                DataSet dataSet = new DataSet(dataSetPath);
                DateTime timePoint = DateTime.Now;
                DataAnalyzer dataAnalyzer = new DataAnalyzer(dataSet, StockDataPath + DSSettings.AnalyzerDataSetsDir, true);
                loadTime += (float)(DateTime.Now - timePoint).TotalMilliseconds;
                dataAnalyzer.SaveDataToFile(StockDataPath + DSSettings.AnalyzerDataSetsDir);
            }

            Console.WriteLine(string.Format("Analyzer time = {0}", loadTime));
            Console.WriteLine();
            //Console.ReadKey();

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
