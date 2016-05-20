using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class DataSetMetaData
    {
        #region Properties

        public string Code { get; set; }

        public string Name { get; set; }

        public string FileName { get; set; }

        public string ID { get; set; }

        public string DataSetFilePath { get; set; }

        public string PriceDataSetFilePath { get; set; }

        public string DataPredictionsFilePath { get; set; }

        public string SimDataPredictionsFilePath { get; set; }

        #endregion
    }

    public class DataSetsMetaData : Dictionary<string, DataSetMetaData>
    {
        #region Properties

        public string CombinedDataPredictionsFilePath { get; set; }

        public string SimCombinedDataPredictionsFilePath { get; set; }

        #endregion

        #region Constructor

        public DataSetsMetaData(string workspace)
        {
            CombinedDataPredictionsFilePath = workspace + DSSettings.PredictionDir + DSSettings.CombinedDataPredictionsFile;
            SimCombinedDataPredictionsFilePath = workspace + DSSettings.SimPredictionDir + DSSettings.CombinedDataPredictionsFile;
            LoadFromFile(workspace + DSSettings.DataSetsMetaDataFile, workspace);
        }

        #endregion

        #region Interface

        public void LoadFromFile(string filePath, string workspace)
        {
            using (StreamReader reader = new StreamReader(filePath))
            {
                while (!reader.EndOfStream)
                {
                    string[] lineData = reader.ReadLine().Split(',');

                    DataSetMetaData metaData = new DataSetMetaData();
                    metaData.Code = lineData[0];
                    metaData.Name = lineData[1];
                    metaData.FileName = lineData[2];
                    metaData.ID = lineData[3];
                    metaData.DataSetFilePath = workspace + DSSettings.DataSetsDir + lineData[2] + ".csv";
                    metaData.PriceDataSetFilePath = workspace + DSSettings.PriceDataSetsDir + lineData[2] + ".csv";
                    metaData.DataPredictionsFilePath = workspace + DSSettings.PredictionDir + lineData[2] + DSSettings.PredictionSuffix + ".csv";
                    metaData.SimDataPredictionsFilePath = workspace + DSSettings.SimPredictionDir + lineData[2] + DSSettings.PredictionSuffix + ".csv";


                    Add(lineData[0], metaData);
                }
            }
        }

        #endregion
    }
}
