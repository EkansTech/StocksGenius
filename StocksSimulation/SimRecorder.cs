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

        public float Profit { get; set; }

        #endregion

        #region Constructor

        public SimRecord(int dataSetRow, DateTime date, float accountBalance, float profit)
        {
            DataSetRow = dataSetRow;
            Date = date;
            AccountBalance = accountBalance;
            Profit = profit;
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

        public float MaxLooseRatio { get; set; }

        #endregion

        #region Constructors

        public SimRecorder(string filePath)
        {
            string fileName = Path.GetFileNameWithoutExtension(filePath);
            string[] fileProperties = fileName.Split('_');
            EffectivePredictionResult = Convert.ToSingle(fileProperties[1]);
            MinProfitRatio = Convert.ToSingle(fileProperties[2]);
            MaxInvestmentsPerStock = Convert.ToInt32(fileProperties[3]);
            MaxNumOfInvestments = Convert.ToInt32(fileProperties[4]);
            MaxLooseRatio = Convert.ToSingle(fileProperties[5]);

            LoadFromFile(filePath);
        }

        public SimRecorder(float effectivePredictionResult, float minProfitRatio, int maxInvestmentsPerStock, int maxNumOfInvestments, float maxLooseRatio)
        {
            EffectivePredictionResult = effectivePredictionResult;
            MinProfitRatio = minProfitRatio;
            MaxInvestmentsPerStock = maxInvestmentsPerStock;
            MaxNumOfInvestments = maxNumOfInvestments;
            MaxLooseRatio = maxLooseRatio;
        }

        #endregion

        #region Interface

        public void AddRecord(int dataSetRow, DateTime date, float accountBalance, float profit)
        {
            Add(new SimRecord(dataSetRow, date, accountBalance, profit));
        }

        public void SaveToFile(string name, string folderPath)
        {
            string filePath = string.Format("{0}\\{1}_{2}_{3}_{4}_{5}_{6}_{7}.csv", folderPath, name, EffectivePredictionResult, MinProfitRatio, MaxInvestmentsPerStock, MaxNumOfInvestments, MaxLooseRatio, DateTime.Now.ToString().Replace(':', '_').Replace('/','_'));

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("DataSet Row, Date, Account Ballance, Profit");

                foreach (SimRecord record in this)
                {
                    writer.WriteLine(string.Format("{0},{1},{2},{3}", record.DataSetRow, record.Date.ToShortDateString(), record.AccountBalance, record.Profit));
                }
            }
        }

        public void LoadFromFile(string filePath)
        {
            using (StreamReader reader = new StreamReader(filePath))
            {
                reader.ReadLine();
                while (!reader.EndOfStream)
                {
                    string lineData = reader.ReadLine();
                    string[] data = lineData.Split(',');
                    Add(new SimRecord(Convert.ToInt32(data[0]), Convert.ToDateTime(data[1]), Convert.ToSingle(data[2]), Convert.ToSingle(data[3])));
                }
            }
        }

        #endregion

    }
}
