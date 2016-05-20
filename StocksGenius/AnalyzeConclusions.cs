using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public class Analyze
    {
        #region Properties

        public string DataSetName { get; set; }

        public StocksData.StocksData StocksData { get; set; }

        public int NumOfPredictions { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public double AverageCorrectness { get; set; }

        public bool IsPositiveInvestment
        {
            get { return DSSettings.PositiveChanges.Contains(PredictedChange.DataItem); }
        }

        public bool IsNegativeInvestment
        {
            get { return DSSettings.NegativeChanges.Contains(PredictedChange.DataItem); }
        }

        #endregion

        #region Constructors

        public Analyze()
        {
            AverageCorrectness = 0;
            DataSetName = string.Empty;
            NumOfPredictions = 0;
        }

        public Analyze(PredictionRecord record)
        {
            AverageCorrectness = record.PredictionCorrectness;
            DataSetName = record.DataSet.DataSetCode;
            NumOfPredictions = 1;
            PredictedChange = record.PredictedChange;
        }

        #endregion

        #region Interface

        public void Update(PredictionRecord record)
        {
            AverageCorrectness = (AverageCorrectness * NumOfPredictions + record.PredictionCorrectness) / (NumOfPredictions + 1);
            NumOfPredictions++;
        }

        #endregion
    }

    public class DataSetAnalyzes : Dictionary<CombinationItem, Analyze>
    {
        #region Properties

        public string DataSetName { get; set; }

        public bool ContainsPositiveInvestmens
        {
            get
            {
                foreach (Analyze analyze in Values)
                {
                    if (analyze.IsPositiveInvestment)
                    {
                        return true;
                    }
                }

                return false;
            }
        }

        public bool ContainsNegativeInvestmens
        {
            get
            {
                foreach (Analyze analyze in Values)
                {
                    if (analyze.IsNegativeInvestment)
                    {
                        return true;
                    }
                }

                return false;
            }
        }

        #endregion

        #region Constructors

        public DataSetAnalyzes(string dataSetName)
        {
            DataSetName = dataSetName;
        }

        #endregion

        #region Interface

        public void Add(CombinationItem combinationItem, PredictionRecord record)
        {
            if (!ContainsKey(record.PredictedChange))
            {
                Add(record.PredictedChange, new Analyze(record));
            }
            else
            {
                this[record.PredictedChange].Update(record);
            }
        }

        #endregion
    }

    public class DailyAnalyzes : Dictionary<string, DataSetAnalyzes>
    {
        #region Interface

        public void Add(string dataSetName, CombinationItem combinationItem, PredictionRecord record)
        {
            if (!ContainsKey(dataSetName))
            {
                Add(dataSetName, new DataSetAnalyzes(dataSetName));
            }

            this[dataSetName].Add(combinationItem, record);
        }

        public void RemoveBadAnalyzes()
        {
            List<string> emptyAnalyzes = new List<string>();
            foreach (string dataSet in Keys)
            {
                List<CombinationItem> badPredictions = new List<CombinationItem>();
                foreach (CombinationItem combinationItem in this[dataSet].Keys)
                {
                    if (combinationItem.Is(DataItem.CloseOpenPositive, 1) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.CloseOpenNegative, 1)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.CloseOpenNegative, 1)); }
                    else if (combinationItem.Is(DataItem.OpenPrevClosePositive, 1) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.OpenPrevCloseNegative, 1)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.OpenPrevCloseNegative, 1)); }
                    else if (combinationItem.Is(DataItem.OpenUp, 3) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.OpenDown, 5)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.OpenDown, 3)); }
                    else if (combinationItem.Is(DataItem.OpenUp, 6) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.OpenDown, 10)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.OpenDown, 6)); }
                    else if (combinationItem.Is(DataItem.OpenUp, 9) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.OpenDown, 20)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.OpenDown, 9)); }
                }

                foreach (CombinationItem combinationItem in badPredictions)
                {
                    this[dataSet].Remove(combinationItem);
                }

                if (this[dataSet].Count == 0)
                {
                    emptyAnalyzes.Add(dataSet);
                }
            }

            foreach (string dataSet in emptyAnalyzes)
            {
                Remove(dataSet);
            }
        }

        #endregion
    }
}
