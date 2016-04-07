using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class BloombergDataSource : StocksDataSource
    {
        public override List<string> GetDataSetsList(string workingDirectory)
        {
            throw new NotImplementedException();
        }
        #region Interface

        public override void LoadAllDataToFolder(string path, string databaseName)
        {
        }

        public override void UpdateDataSets(string workingDirectory)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
