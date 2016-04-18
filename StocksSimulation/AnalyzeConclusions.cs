using StocksData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class AnalyzeComparer : IComparer<Analyze>
    {
        public int Compare(Analyze x, Analyze y)
        {
            if (x.PredictedChange.Range != y.PredictedChange.Range)
            {
                return x.PredictedChange.Range - y.PredictedChange.Range;
            }
            else
            {
                return y.NumOfPredictions - x.NumOfPredictions;
                // return (int)((y.AverageCorrectness * 100) - (x.AverageCorrectness * 100));
            }
        }
    }
    public class Analyze : IComparable
    {
        #region Properties

        public DataSet DataSet { get; set; }

        public int NumOfPredictions { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public double AverageCorrectness { get; set; }

        public string DataSetName { get; }

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
            DataSetName = DataSet.DataSetName;
            NumOfPredictions = 1;
            PredictedChange = record.PredictedChange;
        }

        public Analyze(IEnumerable<PredictionRecord> records)
        {
            AverageCorrectness = records.Average(x => x.PredictionCorrectness);
            DataSet = records.First().DataSet;
            DataSetName = DataSet.DataSetName;
            NumOfPredictions = records.Count();
            PredictedChange = records.First().PredictedChange;
        }

        #endregion

        #region Interface

        public void Update(PredictionRecord record)
        {
            AverageCorrectness = (AverageCorrectness * NumOfPredictions + record.PredictionCorrectness) / (NumOfPredictions + 1);
            NumOfPredictions++;
        }

        public int CompareTo(object obj)
        {
            AnalyzeComparer comparer = new AnalyzeComparer();
            return comparer.Compare(this, obj as Analyze);
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

        #region Constructors

        public DataSetAnalyzes()
        { }

        public DataSetAnalyzes(int capacity) : base(capacity)
        { }

        public DataSetAnalyzes(Dictionary<CombinationItem, Analyze> dataSetAnalyzes) : base(dataSetAnalyzes)
        { }

        public DataSetAnalyzes(IEnumerable<PredictionRecord> records)
        {
            var changeRecords = records.GroupBy(x => x.PredictedChange);
            foreach(CombinationItem prediction in DSSettings.PredictionItems)
            {
                var predictionRecords = records.Where(x => x.PredictedChange.Equals(prediction));
                if (predictionRecords.Count() > 0)
                {
                    Add(prediction, new Analyze(predictionRecords));
                }
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
                    CombinationItem opposite = CombinationItem.Item(DSSettings.OppositeDataItems[combinationItem.DataItem], combinationItem.Range);
                    if (this[dataSet].ContainsKey(opposite) && this[dataSet][combinationItem].CompareTo(this[dataSet][opposite]) > 0)
                    {
                        badPredictions.Add(combinationItem);
                    }
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
