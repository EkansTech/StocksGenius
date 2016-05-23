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

        public DateTime Date { get; set; }

        public double RealMoney { get; set; }

        public double TotalValue { get; set; }

        public int NumOfInvestments { get; set; }

        #endregion

        #region Constructor

        public SimRecord(DateTime date, double realMoney, double totalValue, int numOfInvestments)
        {
            Date = date;
            RealMoney = realMoney;
            TotalValue = totalValue;
            NumOfInvestments = numOfInvestments;
        }

        #endregion
    }

    public class SimRecorder : List<SimRecord>
    {
        #region Properties

        public DateTime StartDate { get; set; }

        public double EffectivePredictionResult { get; set; }

        public double MinProfitRatio { get; set; }

        public int MaxInvestmentsPerStock { get; set; }

        public double MaxLooseRatio { get; set; }

        public byte MinPredictedRange { get; set; }

        public byte MaxPredictedRange { get; set; }

        public int SimulationRun { get; set; }

        public double MinTotalValue { get; set; }

        public double MaxTotalValue { get; set; }

        public int TotalNumOfInvestments { get; set; }

        public double PredictionErrorRange { get; set; }

        public int MinCombinationItemsNum { get; set; }

        public int MaxCombinationItemsNum { get; set; }

        public double MaxTotalValueLoose { get; set; }

        public int NumOfGoodInvestments { get; set; }

        public int SafeDaysNum { get; set; }

        public int MaxNumOfInvestments { get; set; }

        #endregion

        #region Constructors

        public SimRecorder(int simulationRun)
        {
            StartDate = DateTime.MinValue;
            EffectivePredictionResult = 0;
            MinProfitRatio = 0;
            MaxInvestmentsPerStock = 0;
            MaxLooseRatio = 0;
            MinPredictedRange = 0;
            MaxPredictedRange = 0;
            SimulationRun = simulationRun;
            TotalNumOfInvestments = 0;
            MaxTotalValue = 0;
            MinTotalValue = 0;
            PredictionErrorRange = 0;
            MinCombinationItemsNum = 0;
            MaxCombinationItemsNum = 0;
            MaxTotalValueLoose = 0.0;
            NumOfGoodInvestments = 0;
            SafeDaysNum = 0;
            MaxNumOfInvestments = 0;
        }

        public SimRecorder(string filePath)
        {
            string fileName = Path.GetFileNameWithoutExtension(filePath);
            string[] fileProperties = fileName.Split('_');
            StartDate = new DateTime(Convert.ToInt32(fileProperties[0].Split('.')[0]), Convert.ToInt32(fileProperties[0].Split('.')[1]), Convert.ToInt32(fileProperties[0].Split('.')[2]));
            SimulationRun = Convert.ToInt32(fileProperties[1]);
            MinPredictedRange = Convert.ToByte(fileProperties[2]);
            MaxPredictedRange = Convert.ToByte(fileProperties[3]);
            EffectivePredictionResult = Convert.ToDouble(fileProperties[4]);
            MinProfitRatio = Convert.ToDouble(fileProperties[5]);
            MaxLooseRatio = Convert.ToDouble(fileProperties[6]);
            MaxInvestmentsPerStock = Convert.ToInt32(fileProperties[7]);
            MinTotalValue = Convert.ToDouble(fileProperties[8]);
            MaxTotalValue = Convert.ToDouble(fileProperties[9]);
            TotalNumOfInvestments = Convert.ToInt32(fileProperties[10]);
            PredictionErrorRange = Convert.ToDouble(fileProperties[11]);
            MinCombinationItemsNum = Convert.ToInt32(fileProperties[12]);
            MaxCombinationItemsNum = Convert.ToInt32(fileProperties[13]);
            MaxTotalValueLoose = Convert.ToDouble(fileProperties[14]);
            NumOfGoodInvestments = Convert.ToInt32(fileProperties[15]);
            SafeDaysNum = Convert.ToInt32(fileProperties[16]);
            MaxNumOfInvestments = Convert.ToInt32(fileProperties[17]);

            LoadFromFile(filePath);
        }

        public SimRecorder(DateTime startDate, double effectivePredictionResult, double minProfitRatio, int maxInvestmentsPerStock, double maxLooseRatio, byte minPredictedRange, 
            byte maxPredictedRange, int simulationRun, double predictionErrorRange, int minCombinationItemsNum, int maxCombinationItemsNum, int safeDaysNum, int maxNumOfInvestments)
        {
            StartDate = startDate;
            EffectivePredictionResult = effectivePredictionResult;
            MinProfitRatio = minProfitRatio;
            MaxInvestmentsPerStock = maxInvestmentsPerStock;
            MaxLooseRatio = maxLooseRatio;
            MinPredictedRange = minPredictedRange;
            MaxPredictedRange = maxPredictedRange;
            SimulationRun = simulationRun;
            PredictionErrorRange = predictionErrorRange;
            MinCombinationItemsNum = minCombinationItemsNum;
            MaxCombinationItemsNum = maxCombinationItemsNum;
            SafeDaysNum = safeDaysNum;
            MaxNumOfInvestments = maxNumOfInvestments;
        }

        #endregion

        #region Interface

        public void AddRecord(DateTime date, double realMoney, double totalValue, int numOfInvestments)
        {
            Add(new SimRecord(date, realMoney, totalValue, numOfInvestments));
        }

        public void SaveToFile(string folderPath, double maxTotalValue, double minTotalValue, int totalNumOfInvestments, double maxTotalValueLoose, int numOfGoodInvestments)
        {
            if (SimulationRun == 0 && Directory.Exists(folderPath))
            {
                Directory.Delete(folderPath, true);
            }

            Directory.CreateDirectory(folderPath);

            string filePath = string.Format("{0}\\{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{17}_{18}.csv",
                folderPath,
                StartDate.Year + "." + StartDate.Month + "." + StartDate.Day,
                SimulationRun.ToString("0000"),
                MinPredictedRange, 
                MaxPredictedRange, 
                EffectivePredictionResult, 
                MinProfitRatio, 
                MaxLooseRatio, 
                MaxInvestmentsPerStock, 
                minTotalValue, 
                maxTotalValue, 
                totalNumOfInvestments,
                PredictionErrorRange, 
                MinCombinationItemsNum, 
                MaxCombinationItemsNum, 
                maxTotalValueLoose,
                numOfGoodInvestments,
                SafeDaysNum,
                MaxNumOfInvestments);

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("DataSet Row, Date, RealMoney, TotalValue, NumOfInvestments");

                foreach (SimRecord record in this)
                {
                    writer.WriteLine("{0},{1},{2},{3}", record.Date.ToShortDateString(), record.RealMoney, record.TotalValue, record.NumOfInvestments);
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
                    Add(new SimRecord(Convert.ToDateTime(data[0]), Convert.ToDouble(data[1]), Convert.ToDouble(data[2]), Convert.ToInt32(data[3])));
                }
            }
        }

        public static void SaveSummary(string workingDirectory, string fileName)
        {
            List<SimRecorder> recorders = new List<SimRecorder>();
            foreach (string filePath in Directory.GetFiles(workingDirectory + SimSettings.SimulationRecordsDirectory))
            {
                recorders.Add(new SimRecorder(filePath));
            }

            using (StreamWriter writer = new StreamWriter(string.Format("{0}\\{1}{2}.csv", workingDirectory, fileName, DateTime.Now.ToString().Replace(':', '_').Replace('/', '_'))))
            {
                writer.WriteLine("SimulationRun,StartDate,MinCombinationItemsNum,MaxCombinationItemsNum,PredictionErrorRange,MinPredictedRange,MaxPredictedRange,EffectivePredictionResult,MinProfitRatio" +
                    ",MaxInvestmentsPerStock,MaxLooseRatio,SafeDaysNum,MaxNumOfInvestments,MinTotalValue,MaxTotalValue,MaxTotalValueLoose,TotalNumOfInvestments,GoodInvestments,FinalValue,ProfitPerInvestment");
                foreach (SimRecorder recorder in recorders)
                {
                    double profitPerInvestment = (recorder.TotalNumOfInvestments == 0) ? 0 : ((double)recorder.Last().TotalValue - SimSettings.RealMoneyStartValue) / (double)recorder.TotalNumOfInvestments;
                    writer.WriteLine("{0},{1},{2},{3},{4}%,{5},{6},{7},{8}%,{9},{10}%,{11},{12},{13},{14},{15},{16},{17}%,{18},{19}",
                        recorder.SimulationRun,
                        recorder.StartDate.ToShortDateString(),
                        recorder.MinCombinationItemsNum,
                        recorder.MaxCombinationItemsNum,
                        recorder.PredictionErrorRange * 100,
                        recorder.MinPredictedRange,
                        recorder.MaxPredictedRange,
                        recorder.EffectivePredictionResult,
                        recorder.MinProfitRatio * 100,
                        recorder.MaxInvestmentsPerStock,
                        recorder.MaxLooseRatio * 100,
                        recorder.SafeDaysNum,
                        recorder.MaxNumOfInvestments,
                        recorder.MinTotalValue,
                        recorder.MaxTotalValue,
                        recorder.MaxTotalValueLoose,
                        recorder.TotalNumOfInvestments,
                        ((double)recorder.NumOfGoodInvestments / (double)recorder.TotalNumOfInvestments) * 100.0,
                        recorder.Last().TotalValue, 
                        profitPerInvestment);
                }
            }
        }

        #endregion

    }
}
