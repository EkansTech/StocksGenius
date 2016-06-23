using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public abstract class StocksDataSource
    {
        #region Properties



        #endregion

        #region Interface

        public abstract Dictionary<string, string> GetDataSetsList(string workingDirectory);

        public abstract void UpdateDataSets(string workingDirectory, DataSetsMetaData metaData);

        public abstract Dictionary<string, double> GetTodayOpenData(DataSetsMetaData metaData);

        #endregion

        #region Protected Methods

        protected Dictionary<string, string> GetCodesFromFile(string datasetsCodesFilePath)
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
