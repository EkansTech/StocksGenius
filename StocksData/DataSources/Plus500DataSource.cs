using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace StocksData
{
    class Plus500DataSource : StocksDataSource
    {
        public override Dictionary<string, string> GetDataSetsList(string workingDirectory)
        {
            string dataSetCodesFile = workingDirectory + DSSettings.DataSetsMetaDataFile;
            if (!File.Exists(dataSetCodesFile))
            {
                Console.WriteLine("Cannot load data sets list, \"{0}\" file not found", DSSettings.DataSetsMetaDataFile);
                return null;
            }

            return GetCodesFromFile(dataSetCodesFile);
        }

        public override void UpdateDataSets(string workingDirectory, DataSetsMetaData metaData)
        {
            string dataSetDir = workingDirectory + DSSettings.DataSetsDir;
            string dataSetCodesFile = workingDirectory + DSSettings.DataSetsMetaDataFile;
            if (!Directory.Exists(dataSetDir))
            {
                Directory.CreateDirectory(dataSetDir);
            }

            if (!File.Exists(dataSetCodesFile))
            {
                Console.WriteLine("Cannot update data sets, \"{0}\" datasets codes file not found", DSSettings.DataSetsMetaDataFile);
                return;
            }

            foreach (string dataSourceFile in Directory.GetFiles(DSSettings.RootDiretory + DSSettings.DataSetsSourcesDir))
            {
                using (StreamReader reader = new StreamReader(dataSourceFile))
                {
                    while (!reader.EndOfStream)
                    {
                        Dictionary<string, string> dataSetData = ConvertJSONToDictionary(reader.ReadLine());

                        UpdateDataSet(dataSetData, metaData);
                    }
                }
            }

            //metaData.SaveToFile(workingDirectory + DSSettings.DataSetsMetaDataFile, workingDirectory);
        }

        #region Private Methods

        private void UpdateDataSet(Dictionary<string, string> dataSetData, DataSetsMetaData metaData)
        {
            DataSetMetaData dataSetMetaData;

            //dataSetMetaData = metaData.Values.First(x => x.Name == dataSetData["InstrumentName"]);
            dataSetMetaData = metaData.Values.FirstOrDefault(x => x.ID == dataSetData["InstrumentID"]);
            if (dataSetMetaData == null)
            {
                return;
            }
            dataSetMetaData.ID = dataSetData["InstrumentID"];

            if (dataSetMetaData.ID == "84")
            {
                dataSetMetaData.ID = "84";
            }

            DataSet dataSet;

            if (File.Exists(dataSetMetaData.DataSetFilePath))
            {
                dataSet = new DataSet(dataSetMetaData.Code, dataSetMetaData.DataSetFilePath);
            }
            else
            {
                dataSet = new DataSet(dataSetMetaData.Code);
            }

            List<DateTime> dateTime = ReadDateTimesInBase64(dataSetData["DateTime"]);
            List<double> closeRate = ReadDecimalsInBase64(dataSetData["CloseRate"]);
            List<double> highRate = ReadDecimalsInBase64(dataSetData["HighRate"]);
            List<double> lowRate = ReadDecimalsInBase64(dataSetData["LowRate"]);
            List<double> openRate = ReadDecimalsInBase64(dataSetData["OpenRate"]);

            if (dateTime.Count != closeRate.Count || highRate.Count != lowRate.Count || dateTime.Count != openRate.Count || dateTime.Count != highRate.Count || dateTime.Count < 2)
            {
                Console.WriteLine("Error parsing plus 500 {0} corrupted data", dataSetData["InstrumentName"]);
            }

            if (dataSet.DataSetCode == "AIR.PA")
            {
                dataSet.DataSetCode = dataSet.DataSetCode;
            }

            for (int i = dateTime.Count - 1; i >= 0; i--)
            {
                if (dataSet.ContainsTradeDay(dateTime[i].Date))
                {
                    continue;
                }

                List<double> newDateData = new List<double>();

                for (int columnNum = 0; columnNum < (int)DataSet.DataColumns.NumOfColumns; columnNum++)
                {
                    DataSet.DataColumns column = (DataSet.DataColumns)columnNum;
                    switch (column)
                    {
                        case DataSet.DataColumns.Date:
                            newDateData.Add(dateTime[i].Date.Ticks);
                            break;
                        case DataSet.DataColumns.Open:
                            newDateData.Add(openRate[i]);
                            break;
                        case DataSet.DataColumns.High:
                            newDateData.Add(highRate[i]);
                            break;
                        case DataSet.DataColumns.Low:
                            newDateData.Add(lowRate[i]);
                            break;
                        case DataSet.DataColumns.Close:
                            newDateData.Add(closeRate[i]);
                            break;
                        default:
                            newDateData.Add(0.0);
                            break;
                    }
                }

                dataSet.AddRange(newDateData);
            }

            dataSet.SaveDataToFile(dataSetMetaData.DataSetFilePath);
        }

        private string Base64Decode(string b)
        {
            string base64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";

            Regex rgx = new Regex("[^a-zA-Z0-9+/=]");
            b = rgx.Replace(b, "");
            int d, e, h, f, g;
            string c = "";
            for (int j = 0; j < b.Length;)
            {
                d = base64.IndexOf(b[j]);
                j++;
                e = base64.IndexOf(b[j]);
                j++;
                h = base64.IndexOf(b[j]);
                j++;
                f = base64.IndexOf(b[j]);
                j++;
                d = d << 2 | e >> 4;
                e = (e & 15) << 4 | h >> 2;
                g = (h & 3) << 6 | f;
                c += char.ConvertFromUtf32(d); if (h != 64) c += char.ConvertFromUtf32(e); if (f != 64) c += char.ConvertFromUtf32(g);
            }

            return c;
        }

        private List<double> ReadDecimalsInBase64(string b)
        {
            List<double> results = new List<double>();
            b = Base64Decode(b);
            int d, g;
            string c = string.Empty;
            double g1, e;
            for (d = 0; d < b.Length; d += 4)
            {
                for (e = 0, g = 3; g >= 0; g--)
                {
                    e = e * 256 + b[d + g];
                }
                g1 = Math.Floor(e / 1E8);
                e = e % 1E8 * Math.Pow(10, -g1);
                results.Add(e);
            }

            return results;
        }

        private List<DateTime> ReadDateTimesInBase64(string b)
        {
            List<DateTime> results = new List<DateTime>();
            b = Base64Decode(b);
            for (int d = 0; d < b.Length; d += 6)
            {
                double e = 0;
                for (int g = 3; g >= 0; g--)
                {
                    e = e * 256 + b[d + g];
                }
                    e *= 1E3;
                e += 256 * b[d + 5] + b[d + 4];
                DateTime date = new DateTime(1970, 1, 1).AddMilliseconds(e).ToLocalTime();
                results.Add(date);
            }

            return results;
        }

        private Dictionary<string, string> ConvertJSONToDictionary(string message)
        {
            Dictionary<string, string> messageData = new Dictionary<string, string>();
            message = message.Trim('{', '}');
            foreach (string messageElement in message.Split(','))
            {
                if (string.IsNullOrWhiteSpace(messageElement))
                {
                    continue;
                }

                if (!messageElement.Contains(':'))
                {
                    Console.WriteLine("Error in parsing data source for plus 500, no ':'");
                    continue;
                }

                string[] elementData = messageElement.Split(':');
                messageData.Add(elementData[0].Trim('\"'), elementData[1].Trim('\"'));
            }

            return messageData;
        }

        public override Dictionary<string, double> GetTodayOpenData(DataSetsMetaData metaData)
        {
            return null;
        }

        #endregion
    }
}
