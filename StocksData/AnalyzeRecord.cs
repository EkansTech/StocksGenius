using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public enum AnalyzedChange
    {
        Down,
        Up
    }
    
    public class AnalyzeRecord
    {
        #region Properties

        public AnalyzesDataSet.AnalyzeCombination Combination { get; set; }

        public int Depth { get; set; }

        public double PredictionCorrectness { get; set; }

        public AnalyzedChange AnalyzedChange { get; set; } 

        public ChangesDataSet.DataColumns PredictedChange { get; set; }

        #endregion

    }
}
