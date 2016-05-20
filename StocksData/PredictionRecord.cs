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

        public ulong CombinationULong { get; set; }

        public double PredictionCorrectness { get; set; }

        public CombinationItem PredictedChange { get; set; }

        public DataSet DataSet { get; set; }

        public DataPredictions DataPredictions { get; set; }

        #endregion

        #region Constructor

        public PredictionRecord()
        {
        }

        public PredictionRecord(PredictionRecord predictionRecord, DataSet dataSet)
        {
            Combination = predictionRecord.Combination;
            CombinationULong = predictionRecord.CombinationULong;
            PredictionCorrectness = predictionRecord.PredictionCorrectness;
            PredictedChange = predictionRecord.PredictedChange;
            DataSet = dataSet;
            DataPredictions = predictionRecord.DataPredictions;
        }

        #endregion
    }
}
