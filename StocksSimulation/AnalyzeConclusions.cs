using StocksData;
using System;
using System.Collections.Generic;
using System.IO;
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

        public string DataSetName { get; set; }

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

        }

        public Analyze(PredictionRecord record)
        {
            AverageCorrectness = record.PredictionCorrectness;
            DataSet = record.DataSet;
            DataSetName = DataSet.DataSetCode;
            NumOfPredictions = 1;
            PredictedChange = record.PredictedChange;
        }

        public Analyze(IEnumerable<PredictionRecord> records)
        {
            AverageCorrectness = records.Average(x => x.PredictionCorrectness);
            DataSet = records.First().DataSet;
            DataSetName = DataSet.DataSetCode;
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
                    CombinationItem opposite = CombinationItem.Item(DSSettings.OppositeDataItems[combinationItem.DataItem], combinationItem.Range, combinationItem.Offset, combinationItem.ErrorRange);
                    if (this[dataSet].ContainsKey(opposite))// && this[dataSet][combinationItem].CompareTo(this[dataSet][opposite]) > 0)
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

    public class AnalyzesSummary : Dictionary<DateTime, DailyAnalyzes>
    {
        #region Properties

        private string m_FileName = "AnalyzesSummary";

        public string FileName
        {
            get { return m_FileName; }
            set { m_FileName = value; }
        }

        private string m_DirectoryName = "\\AnalyzesSummary\\";

        public string DirectoryName
        {
            get { return m_DirectoryName; }
            set { m_DirectoryName = value; }
        }

        public static string SubDirectory { get; set; }

        public string WorkingDirectory { get; set; }

        public int SimulationRun { get; set; }

        #endregion

        #region Constructors

        public AnalyzesSummary(string workingDirectory, int simulationRun)
        {
            SimulationRun = simulationRun;
            WorkingDirectory = workingDirectory;
        }

        #endregion

        #region Interface

        public void Add(int simulationRun, DateTime day, DailyAnalyzes dailyAnalyzes)
        {
            this.Add(day, dailyAnalyzes);
        }

        public void SaveToFile()
        {
            if (!Directory.Exists(WorkingDirectory + m_DirectoryName))
            {
                Directory.CreateDirectory(WorkingDirectory + m_DirectoryName);
            }

            if (SimulationRun == 0)
            {
                SubDirectory = WorkingDirectory + m_DirectoryName + DateTime.Now.ToString().Replace(':', '_').Replace('/', '_') + "\\";
                Directory.CreateDirectory(SubDirectory);
            }

            string filePath = string.Format("{0}\\{1}_{2}.csv", SubDirectory, FileName, SimulationRun);

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("SimulationRun,SimulationDay,DataSet,DataItem,Range,AverageCorrectness,NumOfPredictions");
                 foreach (DateTime day in this.Keys)
                    {
                    foreach (DataSet dataSet in this[day].Keys.OrderBy(x => x.DataSetCode))
                    {
                        foreach (CombinationItem combinationItem in this[day][dataSet].Keys.OrderBy(x => x.Range))
                        {
                            Analyze analyze = this[day][dataSet][combinationItem];
                            writer.WriteLine("{0},{1},{2},{3},{4},{5}",
                                day,
                                dataSet.DataSetCode,
                                combinationItem.DataItem,
                                combinationItem.Range,
                                analyze.AverageCorrectness,
                                analyze.NumOfPredictions);
                        }
                    }
                }
            }
        }

        #endregion
    }
}
