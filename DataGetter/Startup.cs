using Microsoft.Owin;
using Owin;

[assembly: OwinStartupAttribute(typeof(DataGetter.Startup))]
namespace DataGetter
{
    public partial class Startup {
        public void Configuration(IAppBuilder app) {
            ConfigureAuth(app);
        }
    }
}
