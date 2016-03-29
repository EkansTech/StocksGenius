using StocksData;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    class Program
    {
        static void Main(string[] args)
        {
            string stocksDataPath = "C:\\Ekans\\Stocks\\Quandl\\";
            List<string> dataSetPaths = Directory.GetFiles(stocksDataPath + StocksData.Constants.DataSetsDir).ToList();
            List<string> dataAnalyzerPaths = Directory.GetFiles(stocksDataPath + DataAnalyzer.AnalyzerDataSetsDir).ToList();

            foreach (string dataSetPath in dataSetPaths.Where(x => Path.GetFileName(x).StartsWith("WIKI-AA.csv")))
            {
                string analyzeDataSetPath = dataAnalyzerPaths.First(x => Path.GetFileName(x).StartsWith(Path.GetFileNameWithoutExtension(dataSetPath)));
                DataSet dataSet = new DataSet(dataSetPath);
                DataAnalyzer dataAnalyzer = new DataAnalyzer(analyzeDataSetPath, dataSet);

                AnalyzerSimulator simulation = new AnalyzerSimulator(dataSet, dataAnalyzer);
                simulation.Simulate();
            }
            Console.ReadKey();

            return;

            //List<string> analyzeDataSetPaths = Directory.GetFiles(stocksDataPath + StocksData.Constants.AnalyzesDataSetsDir).ToList();

            //foreach (string dataSetPath in dataSetPaths.Where(x => Path.GetFileName(x).StartsWith("WIKI-AA.csv")))
            //{
            //    string analyzeDataSetPath = analyzeDataSetPaths.First(x => Path.GetFileName(x).StartsWith(Path.GetFileNameWithoutExtension(dataSetPath)));
            //    DataSet dataSet = new DataSet(dataSetPath);
            //    AnalyzesDataSet analyzeDataSet = new AnalyzesDataSet(analyzeDataSetPath);

            //    StocksSimulation simulation = new StocksSimulation(dataSet, analyzeDataSet);
            //    simulation.Simulate();
            //}

            Console.ReadKey();
        }
    }
}
