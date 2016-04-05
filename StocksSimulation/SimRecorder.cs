using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class SimRecord
    {
        #region Properties

        public int DataSetRow { get; set; }

        public DateTime Date { get; set; }

        public float AccountBalance { get; set; }

        #endregion

        #region Constructor

        public SimRecord(int dataSetRow, DateTime date, float accountBalance)
        {
            DataSetRow = dataSetRow;
            Date = date;
            AccountBalance = accountBalance;
        }

        #endregion
    }

    public class SimRecorder : List<SimRecord>
    {
        #region Properties

        public float EffectivePredictionResult { get; set; }

        public float MinProfitRatio { get; set; }

        public int MaxInvestmentsPerStock { get; set; }

        public int MaxNumOfInvestments { get; set; }

        #endregion

        #region Constructors

        public SimRecorder(float effectivePredictionResult, float minProfitRatio, int maxInvestmentsPerStock, int maxNumOfInvestments)
        {
            EffectivePredictionResult = effectivePredictionResult;
            MinProfitRatio = minProfitRatio;
            MaxInvestmentsPerStock = maxInvestmentsPerStock;
            MaxNumOfInvestments = maxNumOfInvestments;
        }

        #endregion

        #region Interface

        public void AddRecord(int dataSetRow, DateTime date, float accountBalance)
        {
            Add(new SimRecord(dataSetRow, date, accountBalance));
        }

        public void SaveToFile(string name, string folderPath)
        {
            string filePath = string.Format("{0}\\{1}_{2}_{3}_{4}_{5}_{6}.csv", folderPath, name, EffectivePredictionResult, MinProfitRatio, MaxInvestmentsPerStock, MaxNumOfInvestments, DateTime.Now.ToString());

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("DataSet Row, Date, Account Ballance");

                foreach (SimRecord record in this)
                {
                    writer.WriteLine(string.Format("{0},{1},{2}", record.DataSetRow, record.Date.ToShortDateString(), record.AccountBalance));
                }
            }
        }

        #endregion

    }
}
