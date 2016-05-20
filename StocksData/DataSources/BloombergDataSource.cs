using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class BloombergDataSource : StocksDataSource
    {
        #region Interface
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
