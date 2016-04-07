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
            string iForexAnalyzerRecordsFolder = "\\iForexAnalyzerRecords\\\\";

            Dictionary<string /*stock name*/, string /*stock dataset file*/> iForexFiles = StocksData.StocksData.LoadStocksListFile(stocksDataPath + iForexStocksListFile);

            AnalyzerSimulator analyzerSimulator = new AnalyzerSimulator(iForexFiles.Values.ToList(), stocksDataPath + DSSettings.DataSetsDir, stocksDataPath + iForexAnalyzerFolder);
            //analyzerSimulator.TestAnalyzeResults(stocksDataPath + iForexTestAnalyzerFolder);
            analyzerSimulator.Simulate();

            //Console.Write(Log.ToString());
            Log.SaveLogToFile(@"C:\Ekans\Stocks\Quandl\AnalyzeSimulator.log");

            List<SimRecorder> recorders = new List<SimRecorder>();
            foreach (string filePath in Directory.GetFiles(stocksDataPath + iForexAnalyzerRecordsFolder))
            {
                recorders.Add(new SimRecorder(filePath));
            }

            using (StreamWriter writer = new StreamWriter(string.Format("{0}\\iForexSimSummary{1}.csv", stocksDataPath, DateTime.Now.ToString().Replace(':', '_').Replace('/', '_'))))
            {
                writer.WriteLine("EffectivePredictionResult, MinProfitRatio, MaxInvestmentsPerStock, MaxNumOfInvestments, MaxLooseRatio, Final Profit");
                foreach (SimRecorder recorder in recorders)
                {
                    writer.WriteLine("{0},{1},{2},{3},{4},{5}", recorder.EffectivePredictionResult, recorder.MinProfitRatio, 
                        recorder.MaxInvestmentsPerStock, recorder.MaxNumOfInvestments, recorder.MaxLooseRatio,recorder.Last().AccountBalance);
                }
            }

            Console.ReadKey();

            return;

        }
    }
}
