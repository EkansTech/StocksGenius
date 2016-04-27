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

        public byte MinPredictedRange { get; set; }

        public byte MaxPredictedRange { get; set; }

        public int SimulationRun { get; set; }

        public double MinTotalProfit { get; set; }

        public double MaxTotalProfit { get; set; }

        #endregion

        #region Constructors

        public SimRecorder(int simulationRun)
        {
            EffectivePredictionResult = 0;
            MinProfitRatio = 0;
            MaxInvestmentsPerStock = 0;
            MaxNumOfInvestments = 0;
            MaxLooseRatio = 0;
            MinPredictedRange = 0;
            MaxPredictedRange = 0;
            SimulationRun = simulationRun;
        }

        public SimRecorder(string filePath)
        {
            string fileName = Path.GetFileNameWithoutExtension(filePath);
            string[] fileProperties = fileName.Split('_');
            SimulationRun = Convert.ToInt32(fileProperties[2]);
            MinPredictedRange = Convert.ToByte(fileProperties[3]);
            MaxPredictedRange = Convert.ToByte(fileProperties[4]);
            EffectivePredictionResult = Convert.ToDouble(fileProperties[5]);
            MinProfitRatio = Convert.ToDouble(fileProperties[6]);
            MaxLooseRatio = Convert.ToDouble(fileProperties[7]);
            MaxInvestmentsPerStock = Convert.ToInt32(fileProperties[8]);
            MaxNumOfInvestments = Convert.ToInt32(fileProperties[9]);
            MinTotalProfit = Convert.ToDouble(fileProperties[10]);
            MaxTotalProfit = Convert.ToDouble(fileProperties[11]);

            LoadFromFile(filePath);
        }

        public SimRecorder(double effectivePredictionResult, double minProfitRatio, int maxInvestmentsPerStock, int maxNumOfInvestments, double maxLooseRatio, byte minPredictedRange, byte maxPredictedRange, int simulationRun)
        {
            EffectivePredictionResult = effectivePredictionResult;
            MinProfitRatio = minProfitRatio;
            MaxInvestmentsPerStock = maxInvestmentsPerStock;
            MaxNumOfInvestments = maxNumOfInvestments;
            MaxLooseRatio = maxLooseRatio;
            MinPredictedRange = minPredictedRange;
            MaxPredictedRange = maxPredictedRange;
            SimulationRun = simulationRun;
        }

        #endregion

        #region Interface

        public void AddRecord(int dataSetRow, DateTime date, double accountBalance, double profit)
        {
            Add(new SimRecord(dataSetRow, date, accountBalance, profit));
        }

        public void SaveToFile(string name, string folderPath, double maxTotalProfit, double minTotalProfit)
        {
            if (SimulationRun == 0 && Directory.Exists(folderPath))
            {
                Directory.Delete(folderPath, true);
            }

            Directory.CreateDirectory(folderPath);

            string filePath = string.Format("{0}\\{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}.csv", folderPath, name, DateTime.Now.ToString().Replace(':', '-').Replace('/', '-'), SimulationRun,
                MinPredictedRange, MaxPredictedRange, EffectivePredictionResult, MinProfitRatio, MaxLooseRatio, MaxInvestmentsPerStock, MaxNumOfInvestments, minTotalProfit, maxTotalProfit);

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
