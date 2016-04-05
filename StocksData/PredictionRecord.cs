using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{

    public class PredictionRecord
    {
        #region Properties
        
        public List<CombinationItem> Combination { get; set; }

        public float PredictionCorrectness { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public DataSet DataSet { get; set; }

        #endregion
    }
}
