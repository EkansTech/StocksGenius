using System;
using System.Collections.Generic;
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

        public abstract List<string> GetDataSetsList(string workingDirectory);

        public abstract void UpdateDataSets(string workingDirectory);

        public abstract void LoadAllDataToFolder(string path, string databaseName);

        #endregion
    }
}
