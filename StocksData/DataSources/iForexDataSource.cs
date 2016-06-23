using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class iForexDataSource : StocksDataSource
    {
        #region Static Interface

        public struct OpenData
        {
            public double OpenValue { get; set; }

            public DateTime Date { get; set; }
        }

        public static Dictionary<string, OpenData> LoadTodayOpenData(string openDataFile, DataSetsMetaData metaData)
        {
            Dictionary<string, OpenData> todayOpenData = new Dictionary<string, OpenData>();
            using (StreamReader reader = new StreamReader(openDataFile))
            {
                while (!reader.EndOfStream)
                {
                    string marketOpenData = reader.ReadLine();

                    if (string.IsNullOrWhiteSpace(marketOpenData))
                    {
                        continue;
                    }

                    marketOpenData = marketOpenData.Substring(marketOpenData.LastIndexOf("[[") + 2);
                    marketOpenData = marketOpenData.Substring(0, marketOpenData.LastIndexOf("]]") - 2);
                    marketOpenData = marketOpenData.Substring(0, marketOpenData.LastIndexOf("]]") - 2);

                    string[] stocksInfo = marketOpenData.Split(new string[] { "],[" }, StringSplitOptions.RemoveEmptyEntries);

                    foreach (string stockInfo in stocksInfo)
                    {
                        string[] stockValues = stockInfo.Split(',');
                        DataSetMetaData dataSetMetaData = metaData.Values.FirstOrDefault(x => x.ID.Equals(stockValues[0]));
                        if (dataSetMetaData == null)
                        {
                            continue;
                        }
                        OpenData openData = new OpenData();
                        openData.OpenValue = Convert.ToDouble(stockValues[3].Trim('\"'));
                        openData.Date = DateTime.Parse(stockValues[6].Trim('\"')).Date;
                        if (!todayOpenData.ContainsKey(dataSetMetaData.Code))
                        {
                            todayOpenData.Add(dataSetMetaData.Code, openData);
                        }
                    }
                }
            }

            return todayOpenData;
        }

        public override Dictionary<string, double> GetTodayOpenData(DataSetsMetaData metaData)
        {
            return null;
        }

        #endregion

        #region Interfaces

        public override Dictionary<string, string> GetDataSetsList(string workingDirectory)
        {
            throw new NotImplementedException();
        }

        public override void UpdateDataSets(string workingDirectory, DataSetsMetaData metaData)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
