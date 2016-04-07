using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;
using Xignite.Sdk.Api;
using Xignite.Sdk.Api.Models.XigniteGlobalHistorical;

namespace StocksData
{
    public class XigniteDataSource : StocksDataSource
    {
        #region Members

        XigniteGlobalHistorical objGlobalHistorical = new XigniteGlobalHistorical("5C6196BB1B4F45C6ADF4F2D2FE38F21E");

        #endregion

        #region Constructors

        public XigniteDataSource()
        {
        }

        public override List<string> GetDataSetsList(string workingDirectory)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region Interface
        public override void LoadAllDataToFolder(string path, string databaseName)
        {
            Dictionary<ExchangeDescription, SymbolList> symbolsDictionary = new Dictionary<ExchangeDescription, SymbolList>();
            Dictionary<ExchangeDescription, Dictionary<string, GlobalHistoricalQuotes>> quotesDictionary = new Dictionary<ExchangeDescription, Dictionary<string, GlobalHistoricalQuotes>>();
            ExchangeList exchanges = objGlobalHistorical.ListExchanges();
            foreach (ExchangeDescription exchange in exchanges.ExchangesDescriptions)
            {
                symbolsDictionary.Add(exchange, objGlobalHistorical.ListSymbols(exchange.MarketIdentificationCode, "A", "Z"));
            }

            foreach (ExchangeDescription exchange in exchanges.ExchangesDescriptions)
            {
                quotesDictionary.Add(exchange, new Dictionary<string, GlobalHistoricalQuotes>());
                foreach (SecurityDescription securityDescription in symbolsDictionary[exchange].SecurityDescriptions)
                {
                    string identifier = securityDescription.Symbol + "." + exchange.MarketIdentificationCode;
                    GlobalHistoricalQuotes historicalQuotes = objGlobalHistorical.GetGlobalHistoricalQuotesRange(identifier, IdentifierTypes.Symbol, AdjustmentMethods.None, "1/1/2000", "");
                    quotesDictionary[exchange].Add(securityDescription.Symbol, historicalQuotes);
                }
            }

            foreach (ExchangeDescription exchange in quotesDictionary.Keys)
            {
                foreach (SecurityDescription securityDescription in symbolsDictionary[exchange].SecurityDescriptions)
                {
                    string identifier = securityDescription.Symbol + "." + exchange.MarketIdentificationCode;
                    using (FileStream fileStream = new FileStream(path + "//" + identifier + ".csv", FileMode.CreateNew))
                    {

                    }
                }
            }
        }

        public override void UpdateDataSets(string workingDirectory)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
