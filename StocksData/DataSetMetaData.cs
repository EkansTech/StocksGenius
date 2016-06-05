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

        public string Workspace { get; set; }

        public string SimPredictionsDir { get; set; }

        public string DataSetFilePath { get { return Workspace + DSSettings.DataSetsDir + FileName; } }

        public string DataPredictionsFilePath { get { return Workspace + DSSettings.PredictionDir + FileName; } }

        public string SimDataPredictionsFilePath { get { return SimPredictionsDir + FileName; } }

        #endregion
    }

    public class DataSetsMetaData : Dictionary<string, DataSetMetaData>
    {
        #region Properties

        public string CombinedDataPredictionsFilePath { get; set; }

        public string SimCombinedDataPredictionsFilePath { get; set; }

        public string SimPredictionDir { get; set; }

        #endregion

        #region Constructor

        public DataSetsMetaData(string workspace)
        {
            CombinedDataPredictionsFilePath = workspace + DSSettings.PredictionDir + DSSettings.CombinedDataPredictionsFile;
            SimCombinedDataPredictionsFilePath = workspace + DSSettings.SimPredictionDir + DSSettings.CombinedDataPredictionsFile;
            LoadFromFile(workspace + DSSettings.DataSetsMetaDataFile, workspace);
            SimPredictionDir = workspace + DSSettings.SimPredictionDir;
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
                    metaData.FileName = lineData[2] + ".csv";
                    metaData.ID = lineData[3];
                    metaData.Workspace = workspace;
                    metaData.SimPredictionsDir = workspace + DSSettings.SimPredictionDir;


                    Add(lineData[0], metaData);
                }
            }
        }

        #endregion
    }
}
