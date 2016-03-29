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

        public abstract void LoadAllDataToFolder(string path, string databaseName);

        #endregion
    }
}
