using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.IO.Compression;
using QuandlCS;
using QuandlCS.Requests;
using QuandlCS.Types;
using QuandlCS.Connection;
using QuandlCS.Helpers;
using QuandlCS.Interfaces;
using System.Net;

namespace StocksData
{
    public class QuandlDataSource : StocksDataSource
    {
        #region Interface

        public override Dictionary<string, string> GetDataSetsList(string workingDirectory)
        {
            string dataSetCodesFile = workingDirectory + DSSettings.DataSetsMetaDataFile;
            if (!File.Exists(dataSetCodesFile))
            {
                Console.WriteLine("Cannot load data sets list, \"{0}\" file not found", DSSettings.DataSetsMetaDataFile);
                return null;
            }

            Dictionary<string, string> dataSetsCodes = GetCodesFromFile(dataSetCodesFile);
            var keys = dataSetsCodes.Keys.ToList();
            foreach(string dataSetName in keys)
            {
                dataSetsCodes[dataSetName] = dataSetsCodes[dataSetName].Replace('/', '-');
            }

            return dataSetsCodes;
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
                Console.WriteLine("Cannot update data sets, \"{0}\" file not found", DSSettings.DataSetsMetaDataFile);
                return;
            }
            
            QuandlDownloadRequest downloadRequest = new QuandlDownloadRequest();
            downloadRequest.APIKey = "U_AZZ4jj9sJpDm1tiRrr";
            downloadRequest.Format = FileFormats.CSV;
            downloadRequest.Frequency = Frequencies.Daily;
            downloadRequest.Truncation = int.MaxValue;
            downloadRequest.Sort = SortOrders.Descending;
            downloadRequest.Transformation = Transformations.None;

            QuandlConnection connection = new QuandlConnection();

            int datasetNumber = 0;
            foreach (string datasetCode in metaData.Keys)
            {
                string dataSetFilePath = metaData[datasetCode].DataSetFilePath;
                bool addOldData = false;
                string oldFileContent = string.Empty;
                if (datasetNumber > 0)
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 2);
                }

                datasetNumber++;

                downloadRequest.Datacode = new Datacode(datasetCode, '/');
                
                Console.Write(new string(' ', Console.WindowWidth));
                Console.SetCursorPosition(0, Console.CursorTop - 1);
                if (File.Exists(dataSetFilePath))
                {
                    Console.WriteLine("Updating dataset: {0}", Path.GetFileName(metaData[datasetCode].DataSetFilePath));
                    using (StreamReader reader = new StreamReader(metaData[datasetCode].DataSetFilePath))
                    {
                        reader.ReadLine();
                        oldFileContent = reader.ReadLine() + Environment.NewLine;
                        DateTime lastDate = Convert.ToDateTime(oldFileContent.Split(',')[0]);
                        oldFileContent += reader.ReadToEnd();
                        downloadRequest.StartDate = lastDate.AddDays(1);
                        addOldData = true;
                    }
                }
                else
                {
                    Console.WriteLine("Creating new dataset: {0}", Path.GetFileName(metaData[datasetCode].DataSetFilePath));
                    downloadRequest.StartDate = DateTime.MinValue;

                }

                Console.WriteLine("Completed {0}%", (((double)datasetNumber) / (double)metaData.Count * 100.0).ToString("0.00"));

                using (StreamWriter writer = new StreamWriter(metaData[datasetCode].DataSetFilePath))
                {
                    writer.Write(connection.Request(downloadRequest));
                    if (addOldData)
                    {
                        writer.Write(oldFileContent);
                    }
                }
            }
        }

        public void LoadAllDataToFolder(string path, string databaseName)
        {
            string datasetsCodesFilePath = DownloadDataSetsCodes(path, databaseName);

            List<string> datasetsCodes = GetCodesFromFile(datasetsCodesFilePath).Values.ToList();

            QuandlDownloadRequest downloadRequest = new QuandlDownloadRequest();
            downloadRequest.APIKey = "U_AZZ4jj9sJpDm1tiRrr";
            downloadRequest.Format = FileFormats.CSV;
            downloadRequest.Frequency = Frequencies.Daily;
            downloadRequest.Truncation = int.MaxValue;
            downloadRequest.Sort = SortOrders.Descending;
            downloadRequest.Transformation = Transformations.None;

            QuandlConnection connection = new QuandlConnection();

            int datasetNumber = 0;
            foreach (string datasetCode in datasetsCodes)
            {
                if (datasetNumber > 0)
                {
                    Console.SetCursorPosition(0, Console.CursorTop - 2);
                }

                datasetNumber++;

                downloadRequest.Datacode = new Datacode(datasetCode, '/');

                Console.Write(new string(' ', Console.WindowWidth));
                Console.SetCursorPosition(0, Console.CursorTop - 1);
                Console.WriteLine("Current Request: {0}", datasetCode.Replace('/', '-') + ".csv");
                Console.WriteLine("Completed {0}%", (((double)datasetNumber) / (double)datasetsCodes.Count * 100.0).ToString("0.00"));

                using (StreamWriter writer = new StreamWriter(path + DSSettings.DataSetsDir + datasetCode.Replace('/', '-') + ".csv"))
                {
                    writer.Write(connection.Request(downloadRequest));
                }
            }
        }

        public override Dictionary<string, double> GetTodayOpenData(DataSetsMetaData metaData)
        {
            return null;
        }

        #endregion

        #region Private Methods

        private string DownloadDataSetsCodes(string path, string databaseName)
        {
            string datasetsCodeFilePath = string.Empty;
            string zipDataCodesPath = path + "WIKI-datasets-codes.zip";
            using (WebClient client = new WebClient())
            {
                client.DownloadFile("https://www.quandl.com/api/v3/databases/" + databaseName + "/codes", zipDataCodesPath);
            }

            using (ZipArchive archive = ZipFile.Open(zipDataCodesPath, ZipArchiveMode.Read))
            {
                foreach (ZipArchiveEntry entry in archive.Entries)
                {
                    if (File.Exists(path + entry.FullName))
                    {
                        File.Delete(path + entry.FullName);
                    }
                }

                archive.ExtractToDirectory(path);

                datasetsCodeFilePath = path + archive.Entries[0].FullName;
            }

            return datasetsCodeFilePath;
        }

        #endregion
    }
}
