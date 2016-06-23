using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.Entity;

namespace DataGetter.Models
{
    class MetadataContext : DbContext
    {
        public MetadataContext() : base("DataGetter")
        {

        }

        public DbSet<MetaData> Metadata { get; set; }   
    }
}
