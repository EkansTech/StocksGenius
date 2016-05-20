using StocksData;
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
            SelectWorkSpace();
            LoadWorkspaceSettings();
            bool exit = false;
            StocksGenius stockGenius = new StocksGenius();

            while (!exit)
            {
                Console.WriteLine("Select an action:");
                Console.WriteLine("1. Update Data Sets");
                Console.WriteLine("2. Build Predictions");
                Console.WriteLine("3. Get Actions");
                Console.WriteLine("4. Simulate");
                Console.WriteLine("5. Simulate Model");
                Console.WriteLine("6. Analyze Predictions");
                Console.WriteLine("7. Run Investor");
                Console.WriteLine("8. Build Sim Predictions");
                Console.WriteLine("9. Run Investor With SimPredictions");
                Console.WriteLine("10. Build Combined Predictions");
                Console.WriteLine("11. Simulate Combined Predictions");
                Console.WriteLine("0. To Exit");

                string input = Console.ReadLine();

                switch (input)
                {
                    case "1": stockGenius.UpdateDataSets(); break;
                    case "2": stockGenius.BuildPredictions(); break;
                    case "3": stockGenius.GetActions(); break;
                    case "4": stockGenius.Simulate(); break;
                    case "5": stockGenius.SimulateModel(); break;
                    case "6": stockGenius.AnalyzePredictions(); break;
                    case "7": stockGenius.RunInvestor(); break;
                    case "8": stockGenius.BuildSimPredictions(); break;
                    case "9": stockGenius.RunInvestor(true); break;
                    case "10": stockGenius.BuildCombinedPredictions(); break;
                    case "11": stockGenius.SimulateCombinedPredictions(); break;
                    case "0": exit = true; break;
                    default:
                        break;
                }
            }
        }

        private static void LoadWorkspaceSettings()
        {
            IniFile settings = new IniFile(SGSettings.Workspace + SGSettings.WorkspaceSettingsIni);
            SGSettings.DataSourceType = ParseDataSourceType(settings.IniReadValue("DataSource", "DataSourceType"));
            SGSettings.DataSetsCodesPrefix = settings.IniReadValue("DataSource", "DataSetsCodesPrefix");

            DSSettings.PredictionErrorRange = settings.IniReadDoubleValue("Prediction", "PredictionErrorRange");
            DSSettings.MinimumChangesForPredictionRatio = settings.IniReadDoubleValue("Prediction", "MinimumChangesForPredictionRatio");
            DSSettings.EffectivePredictionResult = settings.IniReadDoubleValue("Prediction", "EffectivePredictionResult");
            DSSettings.PredictionMaxCombinationSize = settings.IniReadIntValue("Prediction", "PredictionMaxCombinationSize");
            DSSettings.DataRelevantSince = settings.IniReadDateTime("Prediction", "DataRelevantSince");

            SGSettings.EffectivePredictionResult = settings.IniReadDoubleValue("Investment", "EffectivePredictionResult");
            SGSettings.PredictionErrorRange = settings.IniReadDoubleValue("Investment", "PredictionErrorRange");
            SGSettings.MinProfitRatio = settings.IniReadDoubleValue("Investment", "MinProfitRatio");
            SGSettings.MaxLooseRatio = settings.IniReadDoubleValue("Investment", "MaxLooseRatio");
            SGSettings.BuySellPenalty = settings.IniReadDoubleValue("Investment", "BuySellPenalty");
            SGSettings.SafesForStockRate = settings.IniReadDoubleValue("Investment", "SafesForStockRate");
            SGSettings.InvestmentPerStock = settings.IniReadDoubleValue("Investment", "InvestmentPerStock");

            int i = 1;
            string predictionItem;
            while (!string.IsNullOrEmpty(predictionItem = settings.IniReadValue("PredictionItems", string.Format("Item{0}", i))))
            {
                byte range = Convert.ToByte(predictionItem.Split(',')[0].Trim());
                DataItem dataItem = (DataItem)Enum.Parse(typeof(DataItem), predictionItem.Split(',')[1].Trim());
                DSSettings.PredictionItems.Add(new CombinationItem(range, dataItem));
                i++;
            }

            i = 1;
            string changeItem;
            while (!string.IsNullOrEmpty(changeItem = settings.IniReadValue("ChangeItems", string.Format("Item{0}", i))))
            {
                byte range = Convert.ToByte(changeItem.Split(',')[0].Trim());
                DataItem dataItem = (DataItem)Enum.Parse(typeof(DataItem), changeItem.Split(',')[1].Trim());
                DSSettings.ChangeItems.Add(new CombinationItem(range, dataItem));
                i++;
            }
        }

        private static DataSourceTypes ParseDataSourceType(string dataSourceType)
        {
            switch (dataSourceType.ToLower())
            {
                case "quandl": return DataSourceTypes.Quandl;
                case "yahoo": return DataSourceTypes.Yahoo;
                case "xignite": return DataSourceTypes.Xignite;
                case "bloomberg": return DataSourceTypes.Bloomberg;
                default: return DataSourceTypes.Yahoo;
            }
        }

        private static void SelectWorkSpace()
        {
            IniFile stocksSettings = new IniFile(SGSettings.RootDiretory + SGSettings.StockSettingsIni);
            List<string> workspaces = new List<string>();

            int i = 1;

            while (true)
            {
                string workspace = stocksSettings.IniReadValue("Workspaces", "WS" + i.ToString());

                if (string.IsNullOrWhiteSpace(workspace))
                {
                    break;
                }

                workspaces.Add(workspace);
                i++;
            }

            Console.WriteLine("Select Workspace:");
            for (i = 0; i < workspaces.Count; i++)
            {
                Console.WriteLine("{0}. {1}", i, workspaces[i]);
            }

            int workspaceNum = Convert.ToInt32(Console.ReadLine());

            SGSettings.Workspace = SGSettings.RootDiretory + workspaces[workspaceNum] + "\\";
            Console.WriteLine("Selected {0} workspace: {1}", workspaces[workspaceNum], SGSettings.Workspace);
        }
    }
}
