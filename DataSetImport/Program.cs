using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DataSetImport
{
    class Program
    {
        [STAThreadAttribute]
        static void Main(string[] args)
        {
            string[] importFiles = SelectFiles();

            foreach (string file in importFiles)
            {
                ConvertFile(file);
            }
        }

        private static void ConvertFile(string file)
        {
            string fileData = new StreamReader(file).ReadToEnd();
            fileData = fileData.Substring(fileData.IndexOf("[") + 1);
            fileData = fileData.Substring(0, fileData.LastIndexOf("]") - 1);

            var rowsData = fileData.Split('[', ']').Where(x => !string.IsNullOrWhiteSpace(x) && !x.Equals(","));
            using (StreamWriter writer = new StreamWriter(file.Substring(0, file.IndexOf('.')) + ".csv"))
            {
                writer.WriteLine("Date,Open,High,Low,Close,Volume");
                foreach (string row in rowsData.Reverse())
                {
                    string[] rowData = row.Split(',');
                    writer.Write(rowData[0].Trim('\\', '\"'));
                    writer.Write("," + rowData[2]);
                    writer.Write("," + rowData[4]);
                    writer.Write("," + rowData[1]);
                    writer.Write("," + rowData[3]);
                    writer.WriteLine(",0");
                }
            }
        }

        public static string[] SelectFiles()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog()
            {
                Multiselect = true,
                InitialDirectory = @"C:\Ekans\Stocks\iForexImport",
            };

            openFileDialog.ShowDialog();

            return openFileDialog.FileNames;            
        }
    }
}
