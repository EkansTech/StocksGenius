using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public class StockAnalyzer
    {
        #region Enums

        public enum PredictionType
        {
            PositiveUp,
            NegativeUp,
            PositiveDown,
            NegativeDown
        }

        #endregion
        
        #region Members

        Dictionary<string, Dictionary<int, List<float>>> m_StockRelations = new Dictionary<string, Dictionary<int, List<float>>>();

        Dictionary<string, Dictionary<int, Dictionary<PredictionType, List<float>>>> m_RelationsConlusions = new Dictionary<string, Dictionary<int, Dictionary<PredictionType, List<float>>>>();

        Dictionary<string, Dictionary<int, Dictionary<PredictionType, float>>> m_PredictionsCorrectness = new Dictionary<string, Dictionary<int, Dictionary<PredictionType, float>>>();

        #endregion

        #region Properties

        public StockReader StockReader { get; set; }

        #endregion

        #region Constuctors

        public StockAnalyzer(StockReader stockReader)
        {
            StockReader = stockReader;
        }

        #endregion

        #region Interface

        public void BuildRelations(int maxRange = 1, int minRange = 1)
        {
            for (int column = 0; column < StockReader.ColumnNames.Count; column++)
            {
                string columnName = StockReader.ColumnNames[column];
                m_StockRelations.Add(columnName, new Dictionary<int, List<float>>());

                for (int range = minRange; range <= maxRange; range++)
                {
                    m_StockRelations[columnName].Add(range, new List<float>());
                    for (int cycle = 0; cycle < StockReader.DataSize - range - 1; cycle++)
                    {
                        m_StockRelations[columnName][range].Add(CalculateRelation(StockReader.ColumnNames[column], range, cycle));

                    }
                }
            }
        }

        public void CaclulateRelationsConclusions(int history = -1)
        {
            for (int column = 0; column < StockReader.ColumnNames.Count; column++)
            {
                string columnName = StockReader.ColumnNames[column];
                m_RelationsConlusions.Add(columnName, new Dictionary<int, Dictionary<PredictionType, List<float>>>());

                for (int range = m_StockRelations[columnName].Keys.First(); range <= m_StockRelations[columnName].Keys.Last(); range++)
                {
                    m_RelationsConlusions[columnName].Add(range, new Dictionary<PredictionType, List<float>>());
                    m_RelationsConlusions[columnName][range].Add(PredictionType.PositiveUp, CalculatePositiveUp(StockReader.StockDataDelta[columnName], m_StockRelations[columnName][range], history));
                    m_RelationsConlusions[columnName][range].Add(PredictionType.NegativeUp, CalculateNegativeUp(StockReader.StockDataDelta[columnName], m_StockRelations[columnName][range], history));
                    m_RelationsConlusions[columnName][range].Add(PredictionType.PositiveDown, CalculatePositiveDown(StockReader.StockDataDelta[columnName], m_StockRelations[columnName][range], history));
                    m_RelationsConlusions[columnName][range].Add(PredictionType.NegativeDown, CalculateNegativeDown(StockReader.StockDataDelta[columnName], m_StockRelations[columnName][range], history));
                }
            }
        }

        public void CalculatePredictionCorrectness()
        {
            for (int column = 0; column < StockReader.ColumnNames.Count; column++)
            {
                string columnName = StockReader.ColumnNames[column];
                m_PredictionsCorrectness.Add(columnName, new Dictionary<int, Dictionary<PredictionType, float>>());

                for (int range = m_RelationsConlusions[columnName].Keys.First(); range <= m_RelationsConlusions[columnName].Keys.Last(); range++)
                {
                    m_PredictionsCorrectness[columnName].Add(range, new Dictionary<PredictionType, float>());

                    foreach (StockAnalyzer.PredictionType predictionType in typeof(StockAnalyzer.PredictionType).GetEnumValues())
                    {
                        float sum = 0;
                        for (int cycle = 0; cycle < m_RelationsConlusions[columnName][range][predictionType].Count; cycle++)
                        {
                            sum += m_RelationsConlusions[columnName][range][predictionType][cycle];
                        }

                        m_PredictionsCorrectness[columnName][range].Add(predictionType, sum / (float)m_RelationsConlusions[columnName][range][predictionType].Count);
                    }
                }
            }
        }

        public float GetBestPrediction()
        {
            float bestPrediction = 0;
            string bestColumnName = string.Empty;
            int bestRange = 0;
            PredictionType bestPredictionType = PredictionType.NegativeDown;

            for (int column = 0; column < StockReader.ColumnNames.Count; column++)
            {
                string columnName = StockReader.ColumnNames[column];

                if (columnName.Equals("Ex-Dividend") || columnName.Equals("Split Ratio"))
                    continue;

                for (int range = m_PredictionsCorrectness[columnName].Keys.First(); range <= m_PredictionsCorrectness[columnName].Keys.Last(); range++)
                {
                    foreach (PredictionType predictionType in m_PredictionsCorrectness[columnName][range].Keys)
                    {
                        if (m_PredictionsCorrectness[columnName][range][predictionType] > bestPrediction)
                        {
                            bestPrediction = m_PredictionsCorrectness[columnName][range][predictionType];
                            bestColumnName = columnName;
                            bestRange = range;
                            bestPredictionType = predictionType;
                        }            
                    }
                }
            }
            Console.Write(string.Format("Best prediction is {0}%: \nColumn: {1}\nRange: {2}\nPrediction Type: {3}\n",
                bestPrediction * 100, bestColumnName, bestRange, bestPredictionType));

            return bestPrediction;
        }
        public float GetBestPrediction(PredictionType predictionType)
        {
            float bestPrediction = 0;
            string bestColumnName = string.Empty;
            int bestRange = 0;

            for (int column = 0; column < StockReader.ColumnNames.Count; column++)
            {
                string columnName = StockReader.ColumnNames[column];

                if (columnName.Equals("Ex-Dividend") || columnName.Equals("Split Ratio"))
                    continue;

                for (int range = m_PredictionsCorrectness[columnName].Keys.First(); range <= m_PredictionsCorrectness[columnName].Keys.Last(); range++)
                {
                    if (m_PredictionsCorrectness[columnName][range][predictionType] > bestPrediction)
                    {
                        bestPrediction = m_PredictionsCorrectness[columnName][range][predictionType];
                        bestColumnName = columnName;
                        bestRange = range;
                    }
                }
            }
            Console.Write(string.Format("Best prediction is {0}%: \nColumn: {1}\nRange: {2}\n",
                bestPrediction * 100, bestColumnName, bestRange));

            return bestPrediction;
        }

        #endregion

        #region Private Methods

        private float CalculateRelation(string relationColumn, int range, int cycle)
        {
            float sum = 0;

            for (int i = 1; i <= range; i++)
            {
                sum += StockReader.StockDataDelta[relationColumn][cycle + i];
            }

            return sum;
        }

        private List<float> CalculatePositiveUp(List<float> actual, List<float> relation, int history)
        {
            List<float> predictionAccuracies = new List<float>(); 
            if (history == -1)
            {
                history = relation.Count;
            }

            history = (history > relation.Count) ? relation.Count : history;

            for (int cycle = 0; cycle < relation.Count - history; cycle++)
            {
                float predictionAccuracy = 0;
                float countOfPositive = 0;
                for (int i = 0; i < history; i++)
                {
                    if (relation[cycle + i] > 0)
                    {
                        countOfPositive++;
                        predictionAccuracy += (actual[cycle + i] > 0) ? 1 : (actual[cycle + i] < 0) ? -1 : 0;
                    }
                }
                predictionAccuracies.Add(((countOfPositive + predictionAccuracy) / 2.0F) / countOfPositive);
            }

            return predictionAccuracies;
        }

        private List<float> CalculateNegativeUp(List<float> actual, List<float> relation, int history)
        {
            List<float> predictionAccuracies = new List<float>();
            if (history == -1)
            {
                history = relation.Count;
            }

            history = (history > relation.Count) ? relation.Count : history;

            for (int cycle = 0; cycle < relation.Count - history; cycle++)
            {
                float predictionAccuracy = 0;
                float countOfNegatives = 0;
                for (int i = 0; i < history; i++)
                {
                    if (relation[cycle + i] < 0)
                    {
                        countOfNegatives++;
                        predictionAccuracy += (actual[cycle + i] > 0) ? 1 : (actual[cycle + i] < 0) ? -1 : 0;
                    }
                }
                predictionAccuracies.Add(((countOfNegatives + predictionAccuracy) / 2.0F) / countOfNegatives);
            }

            return predictionAccuracies;
        }

        private List<float> CalculatePositiveDown(List<float> actual, List<float> relation, int history)
        {
            List<float> predictionAccuracies = new List<float>();
            if (history == -1)
            {
                history = relation.Count;
            }

            history = (history > relation.Count) ? relation.Count : history;

            for (int cycle = 0; cycle < relation.Count - history; cycle++)
            {
                float predictionAccuracy = 0;
                float countOfPositive = 0;
                for (int i = 0; i < history; i++)
                {
                    if (relation[cycle + i] > 0)
                    {
                        countOfPositive++;
                        predictionAccuracy += (actual[cycle + i] < 0) ? 1 : (actual[cycle + i] > 0) ? -1 : 0;
                    }
                }
                predictionAccuracies.Add(((countOfPositive + predictionAccuracy) / 2.0F) / countOfPositive);
            }

            return predictionAccuracies;
        }

        private List<float> CalculateNegativeDown(List<float> actual, List<float> relation, int history)
        {
            List<float> predictionAccuracies = new List<float>();
            if (history == -1)
            {
                history = relation.Count;
            }

            history = (history > relation.Count) ? relation.Count : history;

            for (int cycle = 0; cycle < relation.Count - history; cycle++)
            {
                float predictionAccuracy = 0;
                float countOfNegatives = 0;
                for (int i = 0; i < history; i++)
                {
                    if (relation[cycle + i] < 0)
                    {
                        countOfNegatives++;
                        predictionAccuracy += (actual[cycle + i] < 0) ? 1 : (actual[cycle + i] > 0) ? -1 : 0;
                    }
                }
                predictionAccuracies.Add(((countOfNegatives + predictionAccuracy) / 2.0F) / countOfNegatives);
            }

            return predictionAccuracies;
        }

        #endregion
    }
}
