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

        public double AccountBalance { get; set; }

        public double Profit { get; set; }

        #endregion

        #region Constructor

        public SimRecord(int dataSetRow, DateTime date, double accountBalance, double profit)
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

        public double EffectivePredictionResult { get; set; }

        public double MinProfitRatio { get; set; }

        public int MaxInvestmentsPerStock { get; set; }

        public int MaxNumOfInvestments { get; set; }

        public double MaxLooseRatio { get; set; }

        public byte MaxPredictedRange { get; set; }

        #endregion

        #region Constructors

        public SimRecorder(string filePath)
        {
            string fileName = Path.GetFileNameWithoutExtension(filePath);
            string[] fileProperties = fileName.Split('_');
            EffectivePredictionResult = Convert.ToDouble(fileProperties[1]);
            MinProfitRatio = Convert.ToDouble(fileProperties[2]);
            MaxInvestmentsPerStock = Convert.ToInt32(fileProperties[3]);
            MaxNumOfInvestments = Convert.ToInt32(fileProperties[4]);
            MaxLooseRatio = Convert.ToDouble(fileProperties[5]);
            MaxPredictedRange = Convert.ToByte(fileProperties[6]);

            LoadFromFile(filePath);
        }

        public SimRecorder(double effectivePredictionResult, double minProfitRatio, int maxInvestmentsPerStock, int maxNumOfInvestments, double maxLooseRatio, byte maxPredictedRange)
        {
            EffectivePredictionResult = effectivePredictionResult;
            MinProfitRatio = minProfitRatio;
            MaxInvestmentsPerStock = maxInvestmentsPerStock;
            MaxNumOfInvestments = maxNumOfInvestments;
            MaxLooseRatio = maxLooseRatio;
            MaxPredictedRange = maxPredictedRange;
        }

        #endregion

        #region Interface

        public void AddRecord(int dataSetRow, DateTime date, double accountBalance, double profit)
        {
            Add(new SimRecord(dataSetRow, date, accountBalance, profit));
        }

        public void SaveToFile(string name, string folderPath)
        {
            string filePath = string.Format("{0}\\{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.csv", folderPath, name, EffectivePredictionResult, MinProfitRatio, MaxInvestmentsPerStock, 
                MaxNumOfInvestments, MaxLooseRatio, MaxPredictedRange, DateTime.Now.ToString().Replace(':', '_').Replace('/','_'));

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
                    Add(new SimRecord(Convert.ToInt32(data[0]), Convert.ToDateTime(data[1]), Convert.ToDouble(data[2]), Convert.ToDouble(data[3])));
                }
            }
        }

        #endregion

    }
}
