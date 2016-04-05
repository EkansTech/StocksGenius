using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xignite.Sdk.Api.Models.XigniteGlobalHistorical;

namespace StocksData
{
    public class Program
    {
        static void Main(string[] args)
        {
            //Alea.CUDA.Settings.Instance.JITCompile.Level = "Diagnostic";
            StocksData stockData = new StocksData("C:\\Ekans\\Stocks\\Quandl\\");
            stockData.BuildiForexAnalyzer();
        }
    }
}
