using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class PredictedItemsAnalyze
    {
        public Dictionary<int, int> AppearencesPerCombinationSize;
        public Dictionary<int, double> AverageAppearencesPerCombinationSize;
        public CombinationItem PredictedItem;
        public double AverageAppearance;
        public double NumOfAppearences;

        public PredictedItemsAnalyze(CombinationItem predictedItem)
        {
            AppearencesPerCombinationSize = new Dictionary<int, int>();
            AverageAppearencesPerCombinationSize = new Dictionary<int, double>();
            for (int i = 0; i < 100; i++)
            {
                AppearencesPerCombinationSize.Add(i, 0);
                AverageAppearencesPerCombinationSize.Add(i, 0);
            }
            AverageAppearance = 0;
            PredictedItem = predictedItem;
            NumOfAppearences = 0;
        }

        public void Update(int combinationSize)
        {
            if (!AppearencesPerCombinationSize.ContainsKey(combinationSize))
            {
                AppearencesPerCombinationSize.Add(combinationSize, 1);
            }
            else
            {
                AppearencesPerCombinationSize[combinationSize]++;
            }
            NumOfAppearences++;
        }

        public void Calculate(int numOfPredictions, Dictionary<int, int> numOfPredictionsPerSize)
        {
            AverageAppearance = NumOfAppearences / numOfPredictions;
            foreach (int size in numOfPredictionsPerSize.Keys)
            {
                if (!AverageAppearencesPerCombinationSize.ContainsKey(size))
                {
                    AverageAppearencesPerCombinationSize.Add(size, AppearencesPerCombinationSize[size] / numOfPredictionsPerSize[size]);
                }
                else
                {
                    AverageAppearencesPerCombinationSize[size] = AppearencesPerCombinationSize[size] / numOfPredictionsPerSize[size];
                }
            }
        }
    }
}
