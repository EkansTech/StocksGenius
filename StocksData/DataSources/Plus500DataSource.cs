using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
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

            int datasetNumber = 0;
            foreach (string datasetCode in metaData.Keys)
            {
                using (WebClient web = new WebClient())
                {
                    bool addOldData = false;
                    string oldFileContent = string.Empty;

                    Console.Write(new string(' ', Console.WindowWidth));
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                    DateTime startDate;
                    if (File.Exists(metaData[datasetCode].DataSetFilePath))
                    {
                        Console.WriteLine("Updating dataset: {0}", Path.GetFileName(metaData[datasetCode].DataSetFilePath));
                        using (StreamReader reader = new StreamReader(metaData[datasetCode].DataSetFilePath))
                        {
                            reader.ReadLine();
                            oldFileContent = reader.ReadLine() + Environment.NewLine;
                            DateTime lastDate = Convert.ToDateTime(oldFileContent.Split(',')[0]);
                            oldFileContent += reader.ReadToEnd();
                            startDate = lastDate.AddDays(1);
                            addOldData = true;
                        }
                    }
                    else
                    {
                        Console.WriteLine("Creating new dataset: {0}", Path.GetFileName(metaData[datasetCode].DataSetFilePath));
                        startDate = new DateTime(1950, 1, 1);

                    }

                    Console.WriteLine("Completed {0}%", (((double)datasetNumber) / (double)metaData.Count * 100.0).ToString("0.00"));
                    datasetNumber++;
                    
                    string newData = string.Empty;
                    try
                    {
                        newData = web.DownloadString(string.Format("http://ichart.finance.yahoo.com/table.csv?s={0}&a={1}&b={2}&c={3}", datasetCode, startDate.Month - 1, startDate.Day, startDate.Year));
                    }
                    catch
                    {
                        Console.WriteLine("No new data available for {0}", datasetCode);
                        continue;
                    }

                    using (StreamWriter writer = new StreamWriter(metaData[datasetCode].DataSetFilePath))
                    {
                        writer.Write(newData);
                        if (addOldData)
                        {
                            writer.Write(oldFileContent);
                        }
                    }

                    Thread.Sleep(1000);
                }
            }
        }

        #region Private Methods

        private Dictionary<string, string> GetCodesFromFile(string datasetsCodesFilePath)
        {
            Dictionary<string, string> datasetsNames = new Dictionary<string, string>();
            StreamReader readStream = new StreamReader(datasetsCodesFilePath);

            while (!readStream.EndOfStream)
            {
                string[] lineData = readStream.ReadLine().Split(',');

                datasetsNames.Add(lineData[1], lineData[0]);
            }

            return datasetsNames;
        }
        #endregion
    }
}
