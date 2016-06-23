using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using DataGetter.Models;
using System.Web.ModelBinding;
using StocksData;
using System.Net;
using System.IO;
using System.Threading;
using System.Net.Http;

namespace DataGetter
{
    public partial class _Default : Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {

        }
        protected void CommandGetData(object sender, EventArgs e)
        {
            string sessionID = TextBoxSessionID.Text;
            string subSessionID = TextBoxSubSessionID.Text;
            DataSetsMetaData metaData = new DataSetsMetaData(DSSettings.Workspace);
            using (HttpClient web = new HttpClient())
            using (StreamWriter writer = new StreamWriter(DSSettings.RootDiretory + DSSettings.DataSetsSourcesDir + DateTime.Now.ToString().Replace("/", "_").Replace(":","_")))
            {
                foreach (string code in metaData.Keys)
                {
                    string data = web.GetStringAsync(string.Format("https://trade.plus500.com/ClientRequest/GetChartDataImm?InstrumentID={0}&FeedResolutionLevel=8&SessionID={1}&SubSessionID={2}", metaData[code].ID, sessionID, subSessionID)).Result;
                    writer.WriteLine(data);
                    Thread.Sleep(2000);
                }
            }
        }

        public IQueryable<DataSetMetaData> GetDataSetMetaData([QueryString("id")] string workspace)
        {
            var _db = new MetadataContext();
            IQueryable<MetaData> query = _db.Metadata;
            if (!string.IsNullOrWhiteSpace(workspace))
            {
                query = query.Where(p => p.Workspace == workspace);
            }
            return query.First().DSMetaData.Values.AsQueryable();
        }

    }
}