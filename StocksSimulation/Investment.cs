using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using StocksData;

namespace StocksSimulation
{
    public class Investment
    {
        #region Properties

        public int Ammount { get; set; }

        public AnalyzerRecord AnalyzerRecord { get; set; }

        public int InvestedDay { get; set; }

        public double Price { get; set; }

        #endregion

    }
}
