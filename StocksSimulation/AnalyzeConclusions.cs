using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class Analyze
    {
        #region Properties

        public DataSet DataSet { get; set; }

        public int NumOfPredictions { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public float AverageCorrectness { get; set; }

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

        public Analyze(PredictionRecord record)
        {
            AverageCorrectness = record.PredictionCorrectness;
            DataSet = record.DataSet;
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

    public class DailyAnalyzes : Dictionary<DataSet, DataSetAnalyzes>
    {
        #region Interface

        public void Add(DataSet dataSet, CombinationItem combinationItem, PredictionRecord record)
        {
            if (!ContainsKey(dataSet))
            {
                Add(dataSet, new DataSetAnalyzes());
            }

            this[dataSet].Add(combinationItem, record);
        }

        public void RemoveBadAnalyzes()
        {
            List<DataSet> emptyAnalyzes = new List<DataSet>();
            foreach (DataSet dataSet in Keys)
            {
                List<CombinationItem> badPredictions = new List<CombinationItem>();
                foreach (CombinationItem combinationItem in this[dataSet].Keys)
                {
                    if (combinationItem.Is(DataItem.CloseOpenDif, 1) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.NegativeCloseOpenDif, 1)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.NegativeCloseOpenDif, 1)); }
                    else if (combinationItem.Is(DataItem.OpenPrevCloseDif, 1) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.NegativeOpenPrevCloseDif, 1)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.NegativeOpenPrevCloseDif, 1)); }
                    else if (combinationItem.Is(DataItem.OpenChange, 3) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.NegativeOpenChange, 5)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.NegativeOpenChange, 3)); }
                    else if (combinationItem.Is(DataItem.OpenChange, 6) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.NegativeOpenChange, 10)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.NegativeOpenChange, 6)); }
                    else if (combinationItem.Is(DataItem.OpenChange, 9) && this[dataSet].ContainsKey(CombinationItem.Item(DataItem.NegativeOpenChange, 20)))
                    { badPredictions.Add(combinationItem); badPredictions.Add(CombinationItem.Item(DataItem.NegativeOpenChange, 9)); }
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

            foreach (DataSet dataSet in emptyAnalyzes)
            {
                Remove(dataSet);
            }
        }

        #endregion
    }
}
