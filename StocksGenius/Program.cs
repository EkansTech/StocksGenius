using StocksData;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace StocksGenius
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            bool exit = false;
            Console.WriteLine("Currently working directory is \"{0}\"", SGSettings.WorkingDirectory);
            StocksGenius stockGenius = new StocksGenius();

            while (!exit)
            {
                Console.WriteLine("Select an action:");
                Console.WriteLine("1. Update Data Sets");
                Console.WriteLine("2. Build Predictions");
                Console.WriteLine("3. Get Actions");
                Console.WriteLine("4. Simulate");
                Console.WriteLine("0. To Exit");

                string input = Console.ReadLine();

                switch (input)
                {
                    case "1": stockGenius.UpdateDataSets(); break;
                    case "2": stockGenius.BuildPredictions(); break;
                    case "3": stockGenius.GetActions(); break;
                    case "4": stockGenius.Simulate(); break;
                    case "0": exit = true; break;
                    default:
                        break;
                }
            }
        }    
    }
}
