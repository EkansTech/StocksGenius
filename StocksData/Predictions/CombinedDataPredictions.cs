using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public class CombinedDataPredictions : DataPredictions
    {
        #region Properties

        private Dictionary<ulong, int> m_Instances = new Dictionary<ulong, int>();

        public Dictionary<ulong, int> Instances
        {
            get { return m_Instances; }
            set { m_Instances = value; }
        }


        #endregion

        #region Constructors

        public CombinedDataPredictions(List<DataPredictions> dataPredictionsList)
        {
            foreach (DataPredictions dataPredictions in dataPredictionsList)
            {
                Add(dataPredictions);
            }
        }

        public CombinedDataPredictions(string filePah)
        {
            LoadFromFile(filePah);
        }

        #endregion

        #region Interface

        public override void LoadFromFile(string filePath, bool loadBadPredictions = false)
        {
            using (StreamReader csvFile = new StreamReader(filePath))
            {

                // Read the first line and validate correctness of columns in the data file
                csvFile.ReadLine();

                while (!csvFile.EndOfStream)
                {
                    Add(csvFile.ReadLine());
                }
            }
        }
        public override void SaveDataToFile(string filePath)
        {
            using (StreamWriter csvFile = new StreamWriter(filePath))
            {
                // Write the first line
                csvFile.WriteLine(GetColumnNamesString());

                foreach (ulong combination in this.Keys)
                {
                    csvFile.WriteLine(GetDataString(combination));
                }
            }
        }

        public override void Add(string dataLine)
        {
            string[] data = dataLine.Split(',');

            ulong combination = Convert.ToUInt64(data[0]);
            int instances = Convert.ToInt32(data[2]);

            List<double> combinationPrediction = new List<double>();

            for (int column = 3; column < data.Length; column++)
            {
                combinationPrediction.Add(Convert.ToDouble(data[column]));
            }

            Add(combination, combinationPrediction);
            m_Instances.Add(combination, instances);
        }

        public void Add(DataPredictions dataPredictions)
        {
            foreach(ulong combination in dataPredictions.Keys)
            {
                if (ContainsKey(combination))
                {
                    for (int i = 0; i < this[combination].Count; i++)
                    {
                        this[combination][i] = (this[combination][i] * m_Instances[combination] + dataPredictions[combination][i]) / (m_Instances[combination] + 1);
                    }
                    m_Instances[combination]++;
                }
                else
                {
                    Add(combination, dataPredictions[combination]);
                    m_Instances.Add(combination, 1);
                }
            }
        }

        public override List<PredictionRecord> GetBestPredictions(double effectivePredictionResult)
        {
            List<PredictionRecord> predictionRecords = new List<PredictionRecord>();
            foreach (ulong combination in Keys)
            {
                for (int dataColumn = 0; dataColumn < NumOfDataColumns; dataColumn++)
                {
                    if (this[combination][dataColumn] >= effectivePredictionResult)
                    {
                        predictionRecords.Add(new PredictionRecord()
                        {
                            CombinationULong = combination,
                            Combination = CombinationItem.ULongToCombinationItems(combination),
                            PredictionCorrectness = this[combination][dataColumn],
                            PredictedChange = DSSettings.PredictionItems[dataColumn],
                            DataPredictions = this,
                        });
                    }
                }
            }

            return predictionRecords.OrderByDescending(x => x.PredictionCorrectness).ToList();
        }

        #endregion

        #region Private Methods

        public bool IsContainsPrediction(DataSet dataSet, CombinationItem combinationItem, int dataRow, double upperErrorBorder, double lowerErrorBorder)
        {
            if (dataSet.NumOfRows <= dataRow + combinationItem.Range * 2)
            {
                return false;
            }
            ChangeMap changeMap = DSSettings.DataItemsCalculationMap[combinationItem.DataItem];

            double change = CalculateChange(dataSet, dataRow, combinationItem.Range, changeMap.FromData, changeMap.OfData, changeMap.FromOffset, changeMap.OfOffset, changeMap.Offset);

            if ((changeMap.IsPositiveChange && change > upperErrorBorder) || (!changeMap.IsPositiveChange && change < lowerErrorBorder))
            {
                return true;
            }
            
            return false;
        }

        public bool IsGoodPrediction(DataSet dataSet, CombinationItem changeItem, CombinationItem predictedItem, int dataRow, double upperErrorBorder, double lowerErrorBorder)
        {
            if (!IsContainsPrediction(changeItem, dataRow + changeItem.Range - 1, upperErrorBorder, lowerErrorBorder))
            {
                return false;
            }

            ChangeMap changeMap = DSSettings.DataItemsCalculationMap[predictedItem.DataItem];

            double fromAverage = CalculateAverage(dataSet, dataRow, predictedItem.Range - 1, changeMap.FromData);
            double ofAverage = CalculateAverage(dataSet, dataRow + predictedItem.Range, predictedItem.Range, changeMap.FromData);
            double change = (fromAverage - ofAverage) / ofAverage;

            if ((changeMap.IsPositiveChange && change <= upperErrorBorder) || (!changeMap.IsPositiveChange && change >= lowerErrorBorder))
            {
                return true;
            }

            return false;
        }

        private double CalculateChange(DataSet dataSet, int dataRow, int range, DataSet.DataColumns dataColumFrom, DataSet.DataColumns dataColumOf, int fromRowOffset, int ofRowOffset, int offset)
        {
            int dataFromStartPosition = fromRowOffset * range + offset;
            int dataOfStartPosition = ofRowOffset * range + offset;
            double sumOf = 0;
            double sumFrom = 0;
            for (int i = dataRow; i < dataRow + range; i++)
            {
                sumOf += dataSet[(i + dataOfStartPosition) * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumOf];
                sumFrom += dataSet[(i + dataFromStartPosition) * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumFrom];
            }

            return (sumFrom - sumOf) / sumOf / range;
        }

        private double CalculateAverage(DataSet dataSet, int dataRow, int range, DataSet.DataColumns dataColum)
        {
            double sum = 0;
            for (int i = dataRow; i < dataRow + range; i++)
            {
                sum += dataSet[i * (int)DataSet.DataColumns.NumOfColumns + (int)dataColum];
            }

            return sum / range;
        }

        private string GetDataString(ulong combination)
        {
            string dataString = combination.ToString() + "," + CombinationItem.CombinationToString(combination) + "," + m_Instances[combination];

            for (int i = 0; i < NumOfDataColumns; i++)
            {
                dataString += "," + this[combination][i].ToString();
            }

            return dataString;
        }

        private string GetColumnNamesString()
        {
            string columnNames = "Combination,CombinationItems,Count";

            for (int changePrediction = 0; changePrediction < DSSettings.PredictionItems.Count; changePrediction++)
            {
                columnNames += "," + DSSettings.PredictionItems[changePrediction].ToString();
            }

            return columnNames;
        }

        #endregion
    }
}
