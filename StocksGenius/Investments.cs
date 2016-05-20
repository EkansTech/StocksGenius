using StocksData;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksGenius
{
    public class Investments : List<Investment>
    {
        #region Properties

        #endregion

        #region Constructors

        public Investments()
        {
        }

        public Investments(Investments investments, InvestmentStatus status)
        {
            if (investments != null)
            {
                AddRange(investments.Where(x => x.Status == status));
            }
        }

        #endregion

        #region Interface

        public void SaveToFile(string fileName)
        {
            string filePath = string.Format("{0}\\{1}", SGSettings.Workspace, fileName);

            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("ID,InvestmentDay,InvestmentType,InvestedPrice,DataSetName,Ammount,PredictedDataItem,PredictedRange,NumOfPredictions,AverageCorrectness,ReleaseDay,ReleasePrice,"
                     + "Profit,InvestmentValue,TotalProfit,StockTotalProfit,AccountBefore,InvestedMoney,ReleaseID,Status");
                foreach (Investment investment in this)
                {
                    writer.WriteLine("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19}",
                        investment.ID,
                        investment.InvestmentDay.ToShortDateString(),
                        investment.InvestmentType,
                        investment.InvestedPrice,
                        investment.DataSetName,
                        investment.Ammount,
                        investment.PredictedChange.DataItem,
                        investment.PredictedChange.Range,
                        investment.Analyze.NumOfPredictions,
                        investment.Analyze.AverageCorrectness,
                        investment.ReleaseDay.ToShortDateString(),
                        investment.ReleasePrice,
                        investment.Profit,
                        investment.InvestmentValue,
                        investment.TotalProfit,
                        investment.StockTotalProfit,
                        investment.AccountBefore,
                        investment.InvestedMoney,
                        investment.ReleaseID,
                        investment.Status);
                }
            }
        }

        public void LoadFromFile(string fileName)
        {
            string filePath = string.Format("{0}\\{1}", SGSettings.Workspace, fileName);

            if (!File.Exists(filePath))
            {
                return;
            }

            using (StreamReader reader = new StreamReader(filePath))
            {
                reader.ReadLine();

                while (!reader.EndOfStream)
                {
                    string[] lineData = reader.ReadLine().Split(',');
                    CombinationItem predictedChange = new CombinationItem(Convert.ToByte(lineData[7]), (DataItem)Enum.Parse(typeof(DataItem), lineData[6]));
                    Analyze analyze = new Analyze()
                    {
                        PredictedChange = predictedChange,
                        AverageCorrectness = Convert.ToDouble(lineData[9]),
                        NumOfPredictions = Convert.ToInt32(lineData[8]),
                        DataSetName = lineData[4]
                    };
                    Investment investment = new Investment(Convert.ToInt32(lineData[0]))
                    {
                        InvestmentDay = Convert.ToDateTime(lineData[1]),
                        InvestmentType = (BuySell)Enum.Parse(typeof(BuySell), lineData[2], true),
                        InvestedPrice = Convert.ToDouble(lineData[3]),
                        DataSetName = lineData[4],
                        Ammount = Convert.ToInt32(lineData[5]),
                        PredictedChange = predictedChange,
                        Analyze = analyze,
                        ReleaseDay = Convert.ToDateTime(lineData[10]),
                        ReleasePrice = Convert.ToDouble(lineData[11]),
                        TotalProfit = Convert.ToDouble(lineData[14]),
                        StockTotalProfit = Convert.ToDouble(lineData[15]),
                        AccountBefore = Convert.ToDouble(lineData[16]),
                        InvestedMoney = Convert.ToDouble(lineData[17]),
                        ReleaseID = Convert.ToInt32(lineData[18]),
                        Status = (InvestmentStatus)Enum.Parse(typeof(InvestmentStatus), lineData[19])
                    };

                    Add(investment);
                }
            }
            if (Count > 0)
            {
                Investment.Reset(this.Max(x => x.ID) + 1, this.Max(x => x.ReleaseID) + 1);
            }
        }

        #endregion
    }
}
