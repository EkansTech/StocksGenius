using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace StocksGenius
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            string[] filePaths = ChooseFiles();

            foreach (string filePath in filePaths)
            {
                Console.Write(string.Format("Analyzing {0} Stocks Data:\n", filePath));

                StockReader stockReader = new StockReader(filePath);
                StockAnalyzer stockAnalyzer = new StockAnalyzer(stockReader);

                stockAnalyzer.BuildRelations(20);

                stockAnalyzer.CaclulateRelationsConclusions(20);

                stockAnalyzer.CalculatePredictionCorrectness();

                foreach (StockAnalyzer.PredictionType predictionType in typeof(StockAnalyzer.PredictionType).GetEnumValues())
                {
                    Console.Write(string.Format("Prediction Type - {0}:\n", predictionType));

                    float bestPrediction = stockAnalyzer.GetBestPrediction(predictionType);
                }
            }
            Console.ReadKey();
        }

        static string[] ChooseFiles()
        {
            FolderBrowserDialog folderDialog = new FolderBrowserDialog()
            {
                SelectedPath = "C:\\Ekans\\Stocks"
            };
            folderDialog.ShowDialog();

            return Directory.GetFiles(folderDialog.SelectedPath, "*.csv");
        }
        
    }
}
