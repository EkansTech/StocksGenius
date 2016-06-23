using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.Entity;
using StocksData;

namespace DataGetter.Models
{
    class MetadataInitializer : DropCreateDatabaseAlways<MetadataContext>
    {
        protected override void Seed(MetadataContext context)
        {
            List<string> workspaces = GetWorkspaces();
            foreach (string workspace in workspaces)
            {
                context.Metadata.Add(new MetaData() { Workspace = workspace });
            }
        }

        public static List<string> GetWorkspaces()
        {
            IniFile stocksSettings = new IniFile(DSSettings.RootDiretory + DSSettings.StockSettingsIni);
            List<string> workspaces = new List<string>();

            int i = 1;

            while (true)
            {
                string workspace = stocksSettings.IniReadValue("Workspaces", "WS" + i.ToString());

                if (string.IsNullOrWhiteSpace(workspace))
                {
                    break;
                }

                workspaces.Add(workspace);
                i++;
            }

            return workspaces;
        }
    }
}
