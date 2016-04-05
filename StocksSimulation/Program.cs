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
            string iForexStocksListFile = "iForexStocks.txt";
            string iForexAnalyzerFolder = "\\iForexAnalyzer\\";
            string iForexTestAnalyzerFolder = "\\iForexTestAnalyzer\\";

            Dictionary<string /*stock name*/, string /*stock dataset file*/> iForexFiles = StocksData.StocksData.LoadStocksListFile(stocksDataPath + iForexStocksListFile);

            AnalyzerSimulator analyzerSimulator = new AnalyzerSimulator(iForexFiles.Values.ToList(), stocksDataPath + DSSettings.DataSetsDir, stocksDataPath + iForexAnalyzerFolder);
            //analyzerSimulator.TestAnalyzeResults(stocksDataPath + iForexTestAnalyzerFolder);
            analyzerSimulator.Simulate();

            //Console.Write(Log.ToString());
            Log.SaveLogToFile(@"C:\Ekans\Stocks\Quandl\AnalyzeSimulator.log");

            Console.ReadKey();

            return;

        }
    }
}
