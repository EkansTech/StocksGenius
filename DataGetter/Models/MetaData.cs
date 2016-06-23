using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.ComponentModel.DataAnnotations;
using StocksData;

namespace DataGetter.Models
{
    public class MetaData
    {
        [Key, Required, StringLength(100), Display(Name = "Workspace")]
        public string Workspace { get; set; }

        public string SessionID { get; set; }

        public string SubSessionID { get; set; }

        public DataSetsMetaData DSMetaData { get; set; }
    }

}