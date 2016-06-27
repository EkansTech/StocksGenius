using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class StockRecord
    {
        #region Properties

        public DateTime Date { get; set; }

        public double RealMoney { get; set; }

        public double TotalValue { get; set; }

        public double TotalProfit { get; set; }

        public int NumOfInvestments { get; set; }

        #endregion

        #region Constructor

        public StockRecord(DateTime date, double realMoney, double totalValue, double totalProfit, int numOfInvestments)
        {
            Date = date;
            RealMoney = realMoney;
            TotalValue = totalValue;
            TotalProfit = totalProfit;
            NumOfInvestments = numOfInvestments;
        }

        #endregion
    }

    public class StockRecorder : List<StockRecord>
    {
        #region Properties

        private string m_FileName = "StockRecorder";

        public string FileName
        {
            get { return m_FileName; }
            set { m_FileName = value; }
        }

        private string m_DirectoryName = "\\StockRecorder\\";

        public string DirectoryName
        {
            get { return m_DirectoryName; }
            set { m_DirectoryName = value; }
        }

        public static string SubDirectory { get; set; }

        public DateTime StartDate { get; set; }

        public double MinChangeForDown { get; set; }

        public double MinProfitRatio { get; set; }

        public int MaxInvestmentsPerStock { get; set; }

        public double MaxLooseLimitRelease { get; set; }

        public byte MinDaysOfUp { get; set; }

        public double MinLimitLooseReopen { get; set; }

        public int SimulationRun { get; set; }

        public double MinTotalProfit { get; set; }

        public double MaxTotalProfit { get; set; }

        public int TotalNumOfInvestments { get; set; }

        public double MinChangeForUp { get; set; }

        public int MinDaysOfDown { get; set; }

        public int MaxDaysUntilProfit { get; set; }

        public double MaxTotalValueLoose { get; set; }

        public int NumOfGoodInvestments { get; set; }

        public int NumOfDayWithoutProift { get; set; }

        public int MaxNumOfInvestments { get; set; }

        public double ChangeErrorRange { get; set; }

        #endregion

        #region Constructors

        public StockRecorder(int simulationRun)
        {
            StartDate = DateTime.MinValue;
            MinChangeForDown = 0;
            MinProfitRatio = 0;
            MaxInvestmentsPerStock = 0;
            MaxLooseLimitRelease = 0;
            MinDaysOfUp = 0;
            MinLimitLooseReopen = 0;
            SimulationRun = simulationRun;
            TotalNumOfInvestments = 0;
            MaxTotalProfit = 0;
            MinTotalProfit = 0;
            MinChangeForUp = 0;
            MinDaysOfDown = 0;
            MaxDaysUntilProfit = 0;
            MaxTotalValueLoose = 0.0;
            NumOfGoodInvestments = 0;
            NumOfDayWithoutProift = 0;
            MaxNumOfInvestments = 0;
            ChangeErrorRange = 0;
        }

        public StockRecorder(string filePath)
        {
            string fileName = Path.GetFileNameWithoutExtension(filePath);
            string[] fileProperties = fileName.Split('_');
            StartDate = new DateTime(Convert.ToInt32(fileProperties[0].Split('.')[0]), Convert.ToInt32(fileProperties[0].Split('.')[1]), Convert.ToInt32(fileProperties[0].Split('.')[2]));
            SimulationRun = Convert.ToInt32(fileProperties[1]);
            MinDaysOfUp = Convert.ToByte(fileProperties[2]);
            MinLimitLooseReopen = Convert.ToDouble(fileProperties[3]);
            MinChangeForDown = Convert.ToDouble(fileProperties[4]);
            MinProfitRatio = Convert.ToDouble(fileProperties[5]);
            MaxLooseLimitRelease = Convert.ToDouble(fileProperties[6]);
            MaxInvestmentsPerStock = Convert.ToInt32(fileProperties[7]);
            MinTotalProfit = Convert.ToDouble(fileProperties[8]);
            MaxTotalProfit = Convert.ToDouble(fileProperties[9]);
            TotalNumOfInvestments = Convert.ToInt32(fileProperties[10]);
            MinChangeForUp = Convert.ToDouble(fileProperties[11]);
            MinDaysOfDown = Convert.ToInt32(fileProperties[12]);
            MaxDaysUntilProfit = Convert.ToInt32(fileProperties[13]);
            MaxTotalValueLoose = Convert.ToDouble(fileProperties[14]);
            NumOfGoodInvestments = Convert.ToInt32(fileProperties[15]);
            NumOfDayWithoutProift = Convert.ToInt32(fileProperties[16]);
            MaxNumOfInvestments = Convert.ToInt32(fileProperties[17]);
            ChangeErrorRange = Convert.ToDouble(fileProperties[18]);

            LoadFromFile(filePath);
        }

        public StockRecorder(DateTime startDate, double effectivePredictionResult, double minProfitRatio, int maxInvestmentsPerStock, double maxLooseLimitRelease, byte minDayOfUp, 
            double minLimitLooseReopen, int simulationRun, double minChangeForUp, int minDayOfDown, int maxCombinationItemsNum, int maxNumOfInvestments, double changeErrorRange)
        {
            StartDate = startDate;
            MinChangeForDown = effectivePredictionResult;
            MinProfitRatio = minProfitRatio;
            MaxInvestmentsPerStock = maxInvestmentsPerStock;
            MaxLooseLimitRelease = maxLooseLimitRelease;
            MinDaysOfUp = minDayOfUp;
            MinLimitLooseReopen = minLimitLooseReopen;
            SimulationRun = simulationRun;
            MinChangeForUp = minChangeForUp;
            MinDaysOfDown = minDayOfDown;
            MaxDaysUntilProfit = maxCombinationItemsNum;
            MaxNumOfInvestments = maxNumOfInvestments;
            ChangeErrorRange = changeErrorRange;
        }

        #endregion

        #region Interface

        public void AddRecord(DateTime date, double realMoney, double totalValue, double totalProfit, int numOfInvestments)
        {
            Add(new StockRecord(date, realMoney, totalValue, totalProfit, numOfInvestments));
        }

        public void SaveToFile(string workingDirectory, double maxTotalValue, double minTotalValue, int totalNumOfInvestments, double maxTotalValueLoose, int numOfGoodInvestments, int numOfDayWithoutProift)
        {
            NumOfDayWithoutProift = numOfDayWithoutProift;
            if (!Directory.Exists(workingDirectory + m_DirectoryName))
            {
                Directory.CreateDirectory(workingDirectory + m_DirectoryName);
            }

            if (SimulationRun == 0)
            {
                SubDirectory = workingDirectory + m_DirectoryName + SimSettings.SimStartTime.ToString().Replace(':', '_').Replace('/', '_') + "\\";
                Directory.CreateDirectory(SubDirectory);
            }

            Directory.CreateDirectory(SubDirectory);

            string filePath = string.Format("{0}\\{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{17}_{18}_{19}.csv",
                SubDirectory,
                StartDate.Year + "." + StartDate.Month + "." + StartDate.Day,
                SimulationRun.ToString("0000"),
                MinDaysOfUp, 
                MinLimitLooseReopen, 
                MinChangeForDown, 
                MinProfitRatio, 
                MaxLooseLimitRelease, 
                MaxInvestmentsPerStock, 
                minTotalValue, 
                maxTotalValue, 
                totalNumOfInvestments,
                MinChangeForUp, 
                MinDaysOfDown, 
                MaxDaysUntilProfit, 
                maxTotalValueLoose,
                numOfGoodInvestments,
                NumOfDayWithoutProift,
                MaxNumOfInvestments,
                ChangeErrorRange);

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("Date, RealMoney, TotalValue, TotalProfit, NumOfInvestments");

                foreach (StockRecord record in this)
                {
                    writer.WriteLine("{0},{1},{2},{3},{4}", record.Date.ToShortDateString(), record.RealMoney, record.TotalValue, record.TotalProfit, record.NumOfInvestments);
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
                    Add(new StockRecord(Convert.ToDateTime(data[0]), Convert.ToDouble(data[1]), Convert.ToDouble(data[2]), Convert.ToDouble(data[3]), Convert.ToInt32(data[4])));
                }
            }
        }

        public static void SaveSummary(string workingDirectory, string fileName)
        {
            List<StockRecorder> recorders = new List<StockRecorder>();
            foreach (string filePath in Directory.GetFiles(SubDirectory))
            {
                recorders.Add(new StockRecorder(filePath));
            }

            using (StreamWriter writer = new StreamWriter(string.Format("{0}\\{1}{2}.csv", workingDirectory, fileName, DateTime.Now.ToString().Replace(':', '_').Replace('/', '_'))))
            {
                writer.WriteLine("SimulationRun,StartDate,MinDayOfDown,MinChangeForDown,MinDaysOfUp,MinChangeForUp,MaxDaysUntilProfit,MinLimitLooseReopen,MinProfitRatio" +
                    ",MaxInvestmentsPerStock,MaxLooseLimitRelease,NumOfDayWithoutProift,ChangeErrorRange,MaxNumOfInvestments,MinTotalProfit,MaxTotalProfit,LooseRatio,TotalNumOfInvestments,GoodInvestments,FinalValue,FinalProfit,ProfitPerInvestment");
                foreach (StockRecorder recorder in recorders)
                {
                    double profitPerInvestment = (recorder.TotalNumOfInvestments == 0) ? 0 : ((double)recorder.Last().TotalValue - SimSettings.RealMoneyStartValue) / (double)recorder.TotalNumOfInvestments;
                    writer.WriteLine("{0},{1},{2},{3}%,{4},{5}%,{6},{7},{8}%,{9},{10}%,{11},{12},{13},{14},{15},{16},{17},{18}%,{19},{20},{21}",
                        recorder.SimulationRun,
                        recorder.StartDate.ToShortDateString(),
                        recorder.MinDaysOfDown,
                        recorder.MinChangeForDown * 100,
                        recorder.MinDaysOfUp,
                        recorder.MinChangeForUp * 100,
                        recorder.MaxDaysUntilProfit,
                        recorder.MinLimitLooseReopen,
                        recorder.MinProfitRatio * 100,
                        recorder.MaxInvestmentsPerStock,
                        recorder.MaxLooseLimitRelease * 100,
                        recorder.NumOfDayWithoutProift,
                        recorder.ChangeErrorRange,
                        recorder.MaxNumOfInvestments,
                        recorder.MinTotalProfit,
                        recorder.MaxTotalProfit,
                        recorder.MaxTotalValueLoose / recorder.MaxTotalProfit,
                        recorder.TotalNumOfInvestments,
                        ((double)recorder.NumOfGoodInvestments / (double)recorder.TotalNumOfInvestments) * 100.0,
                        recorder.Last().TotalValue,
                        recorder.Last().TotalProfit,
                        profitPerInvestment);
                }
            }
        }

        #endregion

    }
}
