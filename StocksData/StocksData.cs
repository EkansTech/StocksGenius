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
            m_DataSetPaths = Directory.GetFiles(m_StocksDataPath + Constants.DataSetsDir).ToList();
        }

        #endregion

        #region Interface

        public void BuildDataSetChanges()
        {
            int datasetNumber = 0;

            Console.WriteLine("Building Changes Data Set");

            foreach (string dataSetPath in m_DataSetPaths)
            {
                if (datasetNumber > 0)
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 2);
                }

                datasetNumber++;

                Console.Write(new string(' ', Console.WindowWidth));
                Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Current Data Set: {0}", Path.GetFileName(dataSetPath));
                Console.WriteLine("Completed {0}%", (((double)datasetNumber) / (double)m_DataSetPaths.Count* 100.0).ToString("0.00"));


                DataSet dataSet = new DataSet(dataSetPath);
                ChangesDataSet dataSetChange = new ChangesDataSet(dataSet);
                dataSetChange.SaveDataToFile(StockDataPath + Constants.ChangeDataSetsDir);
            }
        }

        public void BuildDataSetPredictions()
        {
            int datasetNumber = 0;

            Console.WriteLine("DataSet Analyzer");
            double loadTime = 0;
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
                Console.WriteLine("Completed {0}%", (((double)datasetNumber) / (double)m_DataSetPaths.Count * 100.0).ToString("0.00"));


                DataSet dataSet = new DataSet(dataSetPath);
                DateTime timePoint = DateTime.Now;
                DataAnalyzer dataAnalyzer = new DataAnalyzer(dataSet, true);
                loadTime += (DateTime.Now - timePoint).TotalMilliseconds;
                dataAnalyzer.SaveDataToFile(StockDataPath + DataAnalyzer.AnalyzerDataSetsDir);
            }

            Console.WriteLine(string.Format("Analyzer time = {0}", loadTime));
            Console.WriteLine();
            Console.ReadKey();

            return;

            Console.WriteLine("Building Predictions Data Set");
            loadTime = 0;
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
                Console.WriteLine("Completed {0}%", (((double)datasetNumber) / (double)m_DataSetPaths.Count * 100.0).ToString("0.00"));


                DataSet dataSet = new DataSet(dataSetPath);
                ChangesDataSet changesDataSet = new ChangesDataSet(dataSet, false);
                changesDataSet.SaveDataToFile(StockDataPath + Constants.ChangeDataSetsDir);
                PredictionsDataSet predictionsDataSet = new PredictionsDataSet(changesDataSet, true);
                predictionsDataSet.SaveDataToFile(StockDataPath + Constants.PredictionsDataSetsDir);

                DateTime timePoint = DateTime.Now;
                AnalyzesDataSet analyzesDataSet = new AnalyzesDataSet(changesDataSet, predictionsDataSet, false);
                loadTime += (DateTime.Now - timePoint).TotalMilliseconds;
                analyzesDataSet.SaveDataToFile(StockDataPath + Constants.AnalyzesDataSetsDir);
            }

            Console.WriteLine(string.Format("GPU analyze time = {0}", loadTime));
            Console.WriteLine();
            Console.WriteLine();

            datasetNumber = 0;
            foreach (string dataSetPath in m_DataSetPaths.Where(x => Path.GetFileName(x).StartsWith("WIKI-AA")))
            {
                if (datasetNumber > 0)
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 2);
                }

                datasetNumber++;

                Console.Write(new string(' ', Console.WindowWidth));
                Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Current Data Set: {0}", Path.GetFileName(dataSetPath));
                Console.WriteLine("Completed {0}%", (((double)datasetNumber) / (double)m_DataSetPaths.Count * 100.0).ToString("0.00"));


                DataSet dataSet = new DataSet(dataSetPath);
                ChangesDataSet changesDataSet = new ChangesDataSet(dataSet, false);
                changesDataSet.SaveDataToFile(StockDataPath + Constants.ChangeDataSetsDir);
                PredictionsDataSet predictionsDataSet = new PredictionsDataSet(changesDataSet, true);

                DateTime timePoint = DateTime.Now;
                AnalyzesDataSet analyzesDataSet = new AnalyzesDataSet(changesDataSet, predictionsDataSet, false);
                loadTime += (DateTime.Now - timePoint).TotalMilliseconds;
                analyzesDataSet.SaveDataToFile(StockDataPath + Constants.AnalyzesDataSetsDir);
            }

            Console.WriteLine(string.Format("CPU analyze time = {0}", loadTime));
        }

        #endregion
    }
}
