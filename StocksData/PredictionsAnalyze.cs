using StocksData;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class PredictionsAnalyze
    {
        #region Properties

        private string m_FileName = "PredictionsAnalyze";

        public string FileName
        {
            get { return m_FileName; }
            set { m_FileName = value; }
        }

        private DataPredictions m_DataPredictions;

        public DataPredictions DataPredictions
        {
            get { return m_DataPredictions; }
            set { m_DataPredictions = value; }
        }

        public Dictionary<CombinationItem, CombinationItemAnalyze> CombinationItemsAnalyzes { get; set; }

        public Dictionary<CombinationItem, PredictedItemsAnalyze> PredictedItemsAnalyzes { get; set; }

        #endregion

        #region Constructors

        public PredictionsAnalyze(DataPredictions dataPredictions)
        {
            CombinationItemsAnalyzes = new Dictionary<CombinationItem, CombinationItemAnalyze>();
            foreach (CombinationItem changeItem in DSSettings.ChangeItems)
            {
                CombinationItemsAnalyzes.Add(changeItem, new CombinationItemAnalyze(changeItem));
            }
            PredictedItemsAnalyzes = new Dictionary<CombinationItem, PredictedItemsAnalyze>();
            foreach (CombinationItem predictedItem in DSSettings.PredictionItems)
            {
                PredictedItemsAnalyzes.Add(predictedItem, new PredictedItemsAnalyze(predictedItem));
            }
            DataPredictions = dataPredictions;
            AnalyzePredictions(dataPredictions);
        }

        #endregion

        #region Interface

        public void AnalyzePredictions(DataPredictions dataPredictions)
        {
            Dictionary<int, int> predictionsPerSize = new Dictionary<int, int>();
            foreach (ulong combination in dataPredictions.Keys)
            {
                List<CombinationItem> combinationItems = CombinationItem.ULongToCombinationItems(combination);
                if (!predictionsPerSize.ContainsKey(combinationItems.Count))
                {
                    predictionsPerSize.Add(combinationItems.Count, 1);
                }
                else
                {
                    predictionsPerSize[combinationItems.Count]++;
                }

                foreach (CombinationItem item in combinationItems)
                {
                    CombinationItemsAnalyzes[item].Update(combinationItems.Count);
                }

                for (int i = 0; i < DSSettings.PredictionItems.Count; i++)
                {
                    if (dataPredictions[combination][i] >= DSSettings.EffectivePredictionResult)
                    {
                        PredictedItemsAnalyzes[DSSettings.PredictionItems[i]].Update(combinationItems.Count);
                    }
                }
            }

            foreach (CombinationItem item in CombinationItemsAnalyzes.Keys)
            {
                CombinationItemsAnalyzes[item].Calculate(dataPredictions.Count, predictionsPerSize);
            }

            foreach (CombinationItem item in PredictedItemsAnalyzes.Keys)
            {
                PredictedItemsAnalyzes[item].Calculate(dataPredictions.Count, predictionsPerSize);
            }
        }

        public void SaveToFile()
        {
        }

        public static void AnalyzePredictions(string workingDirectory, DataSetsMetaData metadata)
        {
            Dictionary<string, PredictionsAnalyze> predictionsAnalyzes = new Dictionary<string, PredictionsAnalyze>();
            foreach (string dataSetCode in metadata.Keys)
            {
                DataPredictions dataPredictions = new DataPredictions(metadata[dataSetCode].DataPredictionsFilePath);

                PredictionsAnalyze predictionsAnalyze = new PredictionsAnalyze(dataPredictions);
                predictionsAnalyzes.Add(dataSetCode, predictionsAnalyze);
                predictionsAnalyze.SaveToFile();
            }

            SaveSummary(predictionsAnalyzes, workingDirectory);
        }

        public static void SaveSummary(Dictionary<string, PredictionsAnalyze> predictionAnalyzes, string workingDirectory)
        {
            string filePath = workingDirectory + "CombinationItemsSummary" + DateTime.Now.ToString().Replace(':', '_').Replace('/', '_') + ".csv";
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.Write("DataSet");

                for (int changeItem = 0; changeItem < DSSettings.ChangeItems.Count; changeItem++)
                {
                    writer.Write("," + DSSettings.ChangeItems[changeItem].ToString());
                }

                writer.WriteLine();

                foreach (string dataSetName in predictionAnalyzes.Keys)
                {
                    writer.Write(dataSetName);

                    for (int changeItem = 0; changeItem < DSSettings.ChangeItems.Count; changeItem++)
                    {
                        writer.Write("," + predictionAnalyzes[dataSetName].CombinationItemsAnalyzes[DSSettings.ChangeItems[changeItem]].AverageAppearance);
                    }
                    writer.WriteLine();
                }
            }

            filePath = workingDirectory + "PredictionItemsSummary" + DateTime.Now.ToString().Replace(':', '_').Replace('/', '_') + ".csv";
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.Write("DataSet");

                for (int predictionItem = 0; predictionItem < DSSettings.PredictionItems.Count; predictionItem++)
                {
                    writer.Write("," + DSSettings.PredictionItems[predictionItem].ToString());
                }

                writer.WriteLine();

                foreach (string dataSetName in predictionAnalyzes.Keys)
                {
                    writer.Write(dataSetName);

                    for (int predictionItem = 0; predictionItem < DSSettings.PredictionItems.Count; predictionItem++)
                    {
                        writer.Write("," + predictionAnalyzes[dataSetName].PredictedItemsAnalyzes[DSSettings.PredictionItems[predictionItem]].AverageAppearance);
                    }
                    writer.WriteLine();
                }
            }
        }
    }

    #endregion
}