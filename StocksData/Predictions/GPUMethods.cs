using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea.CUDA;
using Alea.CUDA.IL;
using System.Runtime.InteropServices;
using System.Net.Sockets;
using System.Net;
using System.IO;

namespace StocksData
{
    internal class GPUPredictions : ILGPUModule
    {
        #region Members

        private readonly int m_DataSetWidth = (int)DataSet.DataColumns.NumOfColumns;

        private readonly int m_DataSetNumOfRows;

        private readonly int m_ChangesNum = DSSettings.ChangeItems.Count;

        private readonly int m_PredictionsNum = DSSettings.PredictionItems.Count;

        private readonly int m_MapLength = ChangeMap.NumOfData;

        private DeviceMemory<int> m_DataItemsMapDM;
        private DeviceMemory<double> m_DataSetDM;
        private DeviceMemory<byte> m_PredictionsRangesDM;
        private DeviceMemory<byte> m_ChangesDM;
        private DeviceMemory<byte> m_PredictionsDM;
        private DeviceMemory<byte> m_CombinationItems;
        private DeviceMemory<double> m_PredictionResultsDM;

        #endregion

        #region Constructors

        public GPUPredictions(double[] dataset, int[] changesDataItems, byte[] changesRanges, byte[] changesOffsets, double[] changesErrorRanges, 
            int[] predictionDataItems, byte[] predictionRanges, byte[] predictionOffsets, double[] predictionErrorRanges, int combinationsNum)
            : base(GPUModuleTarget.DefaultWorker)
        {
            bool debugMode = false;
            m_DataSetNumOfRows= dataset.Length / m_DataSetWidth;

            // Allocate memory for initialization
            m_DataItemsMapDM = GPUWorker.Malloc(GetDataItemsMap());
            m_DataSetDM = GPUWorker.Malloc(dataset);
            m_ChangesDM = GPUWorker.Malloc(new byte[m_DataSetNumOfRows * m_ChangesNum]);


            // Initialize
            var block = new dim3(1024);
            var grid = new dim3(m_DataSetNumOfRows / block.x + 1);
            var lp = new LaunchParam(grid, block);

            for (int changeNum = 0; changeNum < DSSettings.ChangeItems.Count; changeNum++)
            {
                GPULaunch(BuildChanges, lp, m_DataSetDM.Ptr, m_ChangesDM.Ptr, m_DataItemsMapDM.Ptr, changesDataItems[changeNum], changesRanges[changeNum],
                    changesOffsets[changeNum], changesErrorRanges[changeNum], changeNum);
            }

            m_PredictionsRangesDM = GPUWorker.Malloc(predictionRanges);
            m_PredictionsDM = GPUWorker.Malloc(new byte[m_DataSetNumOfRows * m_PredictionsNum]);
            for (int predictionNum = 0; predictionNum < DSSettings.PredictionItems.Count; predictionNum++)
            {
                GPULaunch(BuildPredictions, lp, m_DataSetDM.Ptr, m_PredictionsDM.Ptr, m_DataItemsMapDM.Ptr, predictionDataItems[predictionNum], predictionRanges[predictionNum],
                    predictionOffsets[predictionNum], predictionErrorRanges[predictionNum], predictionNum);
            }
            if (debugMode)
            { // debug
                byte[] changes = m_ChangesDM.Gather();
                using (StreamWriter writer = new StreamWriter(@"C:\Ekans\Stocks\changes.csv"))
                {
                    writer.Write("RowNum");
                    foreach (CombinationItem changeItem in DSSettings.ChangeItems)
                    {
                        writer.Write("," + changeItem.ToString());
                    }
                    for (int rowNum = 0; rowNum < m_DataSetNumOfRows; rowNum++)
                    {
                        writer.WriteLine();
                        writer.Write(rowNum);
                        for (int changeNum = 0; changeNum < m_ChangesNum; changeNum++)
                        {
                            writer.Write("," + changes[rowNum * m_ChangesNum + changeNum]);
                        }
                    }
                }
            }

            //Free no more needed memory
            //m_DataItemsMapDM.Dispose();
            //m_DataSetDM.Dispose();

            m_DataItemsMapDM = null;
            m_DataSetDM = null;

            //Allocate memory for predictions caclucaltions
            m_PredictionResultsDM = GPUWorker.Malloc<double>(combinationsNum * m_PredictionsNum);
            m_CombinationItems = GPUWorker.Malloc<byte>(combinationsNum * DSSettings.PredictionMaxCombinationSize);
        }

        #endregion

        #region Interface

        public void FreeGPU()
        {
            //m_PredictionsDataItemsDM.Dispose();
            m_PredictionsRangesDM.Dispose();
            m_ChangesDM.Dispose();
            m_PredictionsDM.Dispose();
            m_PredictionResultsDM.Dispose();

            m_PredictionsRangesDM = null;
            m_ChangesDM = null;
            m_PredictionsDM = null;
            m_PredictionResultsDM = null;
        }

        public double[] PredictCombinations(ulong[] combinations, byte combinationSize, int combinationsNum, int minimumChangesForPrediction, double minimumRelevantPredictionResult)
        {
            //using (var dPredictionsSum = GPUWorker.Malloc<short>(combinationsNum * m_PredictionsNum))
            using (var dCombinations = GPUWorker.Malloc(combinations))
            {
                int numOfThreadsInBlock = 1024;
                int blockX = numOfThreadsInBlock;// / blockY;
                int gridX = combinationsNum / blockX + 1;
                var block = new dim3(blockX);//, blockY);
                var grid = new dim3(gridX);
                var lp = new LaunchParam(grid, block);
                GPULaunch(PredictCombinations, lp, dCombinations.Ptr, combinationSize, m_PredictionResultsDM.Ptr, m_CombinationItems.Ptr, m_ChangesDM.Ptr, m_PredictionsDM.Ptr,
                    m_PredictionsRangesDM.Ptr, combinationsNum, minimumChangesForPrediction, minimumRelevantPredictionResult);//, dPredictionsSum.Ptr);
                return m_PredictionResultsDM.Gather();
            }
        }

        public double[] PredictCombinationsTest(byte[] combinations, byte combinationSize, int combinationsNum, int minimumChangesForPrediction, int numOfRows)
        {
            using (var dCombinations = GPUWorker.Malloc(combinations))
            {
                int numOfThreadsInBlock = 1024;
                int blockY = m_PredictionsNum;
                int blockX = numOfThreadsInBlock / blockY;
                int gridX = combinationsNum / blockX + 1;
                var block = new dim3(blockX, blockY);
                var grid = new dim3(gridX);
                var lp = new LaunchParam(grid, block);
                GPULaunch(PredictCombinationsTest, lp, dCombinations.Ptr, combinationSize, m_PredictionResultsDM.Ptr, m_ChangesDM.Ptr, 
                    m_PredictionsDM.Ptr, m_PredictionsRangesDM.Ptr, combinationsNum, minimumChangesForPrediction, numOfRows);
                return m_PredictionResultsDM.Gather();
            }
        }


        #endregion

        #region Kernel Methods

        [Kernel]
        public void BuildChanges(deviceptr<double> dataSet, deviceptr<byte> changes, deviceptr<int> dataItemsMap, int changeDataItem, byte changeRange,
            byte changeOffset, double changeErrorRange, int changeNum)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;

            deviceptr<byte> currentChanges = changes.Ptr(dataRow * m_ChangesNum);
            deviceptr<double> currentDataSet = dataSet.Ptr(dataRow * m_DataSetWidth);

            int columnFrom = dataItemsMap[changeDataItem * m_MapLength + 0];
            int columnOf = dataItemsMap[changeDataItem * m_MapLength + 1];
            int fromRowOffset = dataItemsMap[changeDataItem * m_MapLength + 2];
            int ofRowOffset = dataItemsMap[changeDataItem * m_MapLength + 3];
            int isPositiveChange = dataItemsMap[changeDataItem * m_MapLength + 4];
            int offset = dataItemsMap[changeDataItem * m_MapLength + 5];

            currentChanges[changeNum] = (dataRow >= m_DataSetNumOfRows - (changeRange * 2 + changeOffset)) ?
                    (byte)0 : IsChange(currentDataSet, changeRange, changeOffset, columnFrom, columnOf, fromRowOffset, ofRowOffset, isPositiveChange, changeErrorRange, -changeErrorRange, offset);

        }

        [Kernel]
        public void BuildPredictions(deviceptr<double> dataSet, deviceptr<byte> predictions, deviceptr<int> dataItemsMap, int predictionDataItem, byte predictionRange,
            byte predictionOffset, double predictionErrorRange, int predictionNum)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentPredictions = predictions.Ptr(dataRow * m_PredictionsNum);
            deviceptr<double> currentDataSet = dataSet.Ptr(dataRow * m_DataSetWidth);

                int columnFrom = dataItemsMap[predictionDataItem * m_MapLength + 0];
                int columnOf = dataItemsMap[predictionDataItem * m_MapLength + 1];
                int fromRowOffset = dataItemsMap[predictionDataItem * m_MapLength + 2];
                int ofRowOffset = dataItemsMap[predictionDataItem * m_MapLength + 3];
                int isPositiveChange = dataItemsMap[predictionDataItem * m_MapLength + 4];

                currentPredictions[predictionNum] = (dataRow <= (predictionRange + predictionOffset)) ?
                    (byte)0 : IsPrediction(currentDataSet, predictionRange, predictionOffset, columnFrom, columnOf, fromRowOffset, ofRowOffset, isPositiveChange, predictionErrorRange, -predictionErrorRange);
        }

        [Kernel]
        public void PredictCombinations(deviceptr<ulong> combinations, byte combinationSize, deviceptr<double> predictionResults, deviceptr<byte> combinationItems, deviceptr<byte> changes, 
            deviceptr<byte> predictions, deviceptr<byte> predictionRanges, int combinationsNum, int minimumChangesForPrediction, double minimumRelevantPredictionResult)
        {
            var combinationNum = blockIdx.x * blockDim.x + threadIdx.x;

            if (combinationNum < combinationsNum)
            {
                var currentCombinationItems = combinationItems.Ptr(combinationNum * DSSettings.PredictionMaxCombinationSize);
                byte numOfItems = 0;

                for (byte i = 0; i < m_ChangesNum && numOfItems < combinationSize; i++)
                {
                    if ((combinations[combinationNum] & (((ulong)1) << i)) != 0)
                    {
                        currentCombinationItems[numOfItems] = i;
                        numOfItems++;
                        if (numOfItems == combinationSize)
                        {
                            break;
                        }
                    }
                }

                //deviceptr<short> threadPredictionsSum = predictionsSum.Ptr(combinationNum * m_PredictionsNum);
                var threadPredictionsSum = __local__.Array<short>(m_PredictionsNum);

                for (byte predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
                {
                    threadPredictionsSum[predictionNum] = 0;
                }

                short changesSum = 0;
                for (int rowNum = 0; rowNum < m_DataSetNumOfRows; rowNum++)
                {
                    short change = 1;
                    for (byte itemNum = 0; itemNum < combinationSize; itemNum++)
                    {
                        change *= changes[rowNum * m_ChangesNum + currentCombinationItems[itemNum]];
                    }

                    changesSum += change;
                    for (byte predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
                    {
                        threadPredictionsSum[predictionNum] += (short)(change * predictions[rowNum * m_PredictionsNum + predictionNum]);
                    }
                }

                for (byte predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
                {
                    bool isRelevant = (changesSum > minimumChangesForPrediction)
                        && (double)threadPredictionsSum[predictionNum] / minimumChangesForPrediction >= minimumRelevantPredictionResult;
                    double predictionResult = isRelevant ? (double)threadPredictionsSum[predictionNum] / (double)changesSum : 0.0;
                    predictionResults[combinationNum * m_PredictionsNum + predictionNum] = predictionResult;
                }
            }
        }

        [Kernel]
        public void PredictCombinationsTest(deviceptr<byte> combinationItems, byte combinationSize, deviceptr<double> predictionResults, deviceptr<byte> predictedChanges, 
            deviceptr<byte> actualChanges, deviceptr<byte> predictionRanges, int combinationsNum, int minimumChangesForPrediction, int numOfRows)
        {
            var combinationNum = blockIdx.x * blockDim.x + threadIdx.x;
            var predictionNum = threadIdx.y;

            if (combinationNum < combinationsNum)
            {
                deviceptr<byte> threadCombinationItems = combinationItems.Ptr(combinationNum * combinationSize);

                double predictedChangesSum = 0;
                double actualChangesSum = 0;
                for (int rowNum = 0; rowNum < numOfRows; rowNum++)
                {
                    int predictedChange = 1;
                    for (int itemNum = 0; itemNum < combinationSize; itemNum++)
                    {
                        predictedChange *= predictedChanges[rowNum * m_ChangesNum + threadCombinationItems[itemNum]];
                    }

                    predictedChangesSum += predictedChange;
                    actualChangesSum += predictedChange * actualChanges[rowNum * m_PredictionsNum + predictionNum];
                }
                double predictionResult = (predictedChangesSum > minimumChangesForPrediction) ? actualChangesSum / predictedChangesSum : 0.0;
                predictionResults[combinationNum * m_PredictionsNum + predictionNum] = predictionResult;
            }
        }

        private byte IsPrediction(deviceptr<double> dataSet, byte predictionRange, byte predictionOffset, int dataColumFrom, int dataColumOf, int fromRowOffset, int ofRowOffset, int isPositiveChange, 
            double biggerErrorBorder, double smallerErrorBorder)
        {
            int ofRow = ofRowOffset * predictionRange + predictionOffset;
            int fromRow = fromRowOffset * predictionRange - predictionRange + predictionOffset;
            double sumOf = 0;
            double sumFrom = 0;
            for (int i = 0; i < predictionRange; i++)
            {
                sumOf += dataSet[(ofRow + i) * m_DataSetWidth + dataColumOf];
                sumFrom += dataSet[(fromRow + i) * m_DataSetWidth + dataColumFrom];
            }

            return (byte)((isPositiveChange == 1) ?
                (((sumFrom - sumOf) / sumOf / predictionRange) >= biggerErrorBorder) ? 1 : 0
                :
                (((sumFrom - sumOf) / sumOf / predictionRange) <= smallerErrorBorder) ? 1 : 0);
        }

        private byte IsChange(deviceptr<double> dataSet, byte changeRange, byte changeOffset, int dataColumFrom, int dataColumOf, int fromRowOffset, int ofRowOffset, int isPositiveChange, double biggerErrorBorder,
            double smallerErrorBorder, int offset)
        {
            int ofRow = ofRowOffset * changeRange + offset + changeOffset;
            int fromRow = fromRowOffset * changeRange + offset + changeOffset;
            double sumOf = 0;
            double sumFrom = 0;
            for (int i = 0; i < changeRange; i++)
            {
                sumOf += dataSet[(ofRow + i) * m_DataSetWidth + dataColumOf];
                sumFrom += dataSet[(fromRow + i) * m_DataSetWidth + dataColumFrom];
            }

            return (byte)((isPositiveChange == 1) ?
                (((sumFrom - sumOf) / sumOf / changeRange) >= biggerErrorBorder) ? 1 : 0
                :
                (((sumFrom - sumOf) / sumOf / changeRange) <= smallerErrorBorder) ? 1 : 0);
        }

        private int[] GetDataItemsMap()
        {
            int[] dataItemsMap = new int[DSSettings.DataItems.Count * m_MapLength];

            for (int i = 0; i < DSSettings.DataItems.Count; i++)
            {
                dataItemsMap[i * m_MapLength + 0] = (int)DSSettings.DataItemsCalculationMap[DSSettings.DataItems[i]].FromData;
                dataItemsMap[i * m_MapLength + 1] = (int)DSSettings.DataItemsCalculationMap[DSSettings.DataItems[i]].OfData;
                dataItemsMap[i * m_MapLength + 2] = DSSettings.DataItemsCalculationMap[DSSettings.DataItems[i]].FromOffset;
                dataItemsMap[i * m_MapLength + 3] = DSSettings.DataItemsCalculationMap[DSSettings.DataItems[i]].OfOffset;
                dataItemsMap[i * m_MapLength + 4] = DSSettings.DataItemsCalculationMap[DSSettings.DataItems[i]].IsPositiveChange ? 1 : 0;
                dataItemsMap[i * m_MapLength + 5] = DSSettings.DataItemsCalculationMap[DSSettings.DataItems[i]].Offset;
            }

            return dataItemsMap;
        }

        #endregion
    }

    //public class GPUSimulator : ILGPUModule
    //{
    //    #region Members

    //    private readonly int m_DataSetWidth = (int)DataSet.DataColumns.NumOfColumns;

    //    private readonly int m_ChangesNum = DSSettings.ChangeItems.Count;

    //    private readonly int m_PredictionsNum = DSSettings.PredictionItems.Count;

    //    private readonly double m_ErrorRange = DSSettings.PredictionErrorRange;

    //    private readonly int m_MapLength = DSSettings.ChangeMap.NumOfData;

    //    private DeviceMemory<int> m_DataItemsMapDM;
    //    private DeviceMemory<double> m_DataSetDM;
    //    private DeviceMemory<int> m_ChangesDataItemsDM;
    //    private DeviceMemory<byte> m_ChangesRangesDM;
    //    private DeviceMemory<int> m_PredictionsDataItemsDM;
    //    private DeviceMemory<byte> m_PredictionsRangesDM;
    //    private DeviceMemory<byte> m_ChangesDM;
    //    private DeviceMemory<byte> m_PredictionsDM;
    //    private DeviceMemory<double> m_PredictionResultsDM;

    //    #endregion

    //    #region Constructors

    //    public GPUSimulator() : base(GPUModuleTarget.DefaultWorker)
    //    {
    //    }

    //    #endregion

    //    #region Interface

    //    private bool[] CalcPredictionFits(int dataSetRow, PredictionRecord predictionRecord)
    //    {


    //    }


    //    #endregion

    //    #region Kernel Methods

    //    [Kernel]
    //    public void CalcPredictionFits(int dataRow, deviceptr<ulong> combinations, deviceptr<predic deviceptr<bool> predictionFitResults, deviceptr<double> dataSet, deviceptr<int> changesDataItems, deviceptr<byte> changesRanges,
    //        int combinationsNum, int minimumChangesForPrediction, double minimumRelevantPredictionResult, deviceptr<int> dataItemsMap, double upperErrorBorder, double lowerErrorBorder)
    //    {
    //        var combinationNum = blockIdx.x * blockDim.x + threadIdx.x;
    //        ulong combination = combinations[combinationNum];

    //        if (combinationNum < combinationsNum)
    //        {
    //            int columnFrom = dataItemsMap[predictionDataItem * m_MapLength + 0];
    //            int columnOf = dataItemsMap[predictionDataItem * m_MapLength + 1];
    //            int fromRowOffset = dataItemsMap[predictionDataItem * m_MapLength + 2];
    //            int ofRowOffset = dataItemsMap[predictionDataItem * m_MapLength + 3];
    //            int isPositiveChange = dataItemsMap[predictionDataItem * m_MapLength + 4];
    //            int range = predictionDataItem;

    //            double fromAverage = CalculateAverage(dataRow, predictedItem.Range - 1, changeMap.FromData);
    //            double ofAverage = CalculateAverage(dataRow + predictedItem.Range, predictedItem.Range, changeMap.FromData);
    //            double change = (fromAverage - ofAverage) / ofAverage;

    //            predictionFitResults[combinationNum] = (changeMap.IsPositiveChange && change <= upperErrorBorder) || (!changeMap.IsPositiveChange && change >= lowerErrorBorder);

    //            if (predictionFitResults[combinationNum])
    //            {
    //                ulong combinationItem = 1;
    //                for (int changeNum = 0; changeNum < m_ChangesNum; changeNum++, combinationItem *= 2)
    //                {
    //                    if ((combinationItem & combination) == 0)
    //                    {
    //                        continue;
    //                    }

    //                    columnFrom = dataItemsMap[changesDataItems[changeNum] * m_MapLength + 0];
    //                    columnOf = dataItemsMap[changesDataItems[changeNum] * m_MapLength + 1];
    //                    fromRowOffset = dataItemsMap[changesDataItems[changeNum] * m_MapLength + 2];
    //                    ofRowOffset = dataItemsMap[changesDataItems[changeNum] * m_MapLength + 3];
    //                    isPositiveChange = dataItemsMap[changesDataItems[changeNum] * m_MapLength + 4];
    //                    range = changesRanges[changeNum];

    //                    if (IsPrediction(dataSet, dataRow + range - 1, range, columnFrom, columnOf, fromRowOffset, ofRowOffset, isPositiveChange, upperErrorBorder, lowerErrorBorder) == 0)
    //                    {
    //                        predictionFitResults[combinationNum] = false;
    //                        return;
    //                    }
    //                }
    //            }
    //        }
    //    }

    //    private double CalculateAverage(deviceptr<double> dataSet, int dataRow, int range, int dataColum)
    //    {
    //        double sum = 0;
    //        for (int i = dataRow; i < dataRow + range; i++)
    //        {
    //            sum += dataSet[i * (int)DataSet.DataColumns.NumOfColumns + (int)dataColum];
    //        }

    //        return sum / range;
    //    }

    //    private byte IsPrediction(deviceptr<double> dataSet, int dataRow, int range, int dataColumFrom, int dataColumOf,
    //        int fromRowOffset, int ofRowOffset, int isPositiveChange, double biggerErrorBorder, double smallerErrorBorder)
    //    {
    //        int ofRow = ofRowOffset * range;
    //        int fromRow = fromRowOffset * range;
    //        double sumOf = 0;
    //        double sumFrom = 0;
    //        for (int i = dataRow; i < dataRow + range; i++)
    //        {
    //            sumOf += dataSet[(ofRow + i) * m_DataSetWidth + dataColumOf];
    //            sumFrom += dataSet[(fromRow + i) * m_DataSetWidth + dataColumFrom];
    //        }

    //        return (byte)((isPositiveChange == 1) ?
    //            (((sumFrom - sumOf) / sumOf / range) > biggerErrorBorder) ? 1 : 0
    //            :
    //            (((sumFrom - sumOf) / sumOf / range) < smallerErrorBorder) ? 1 : 0);
    //    }

    //    private int[] GetDataItemsMap()
    //    {
    //        int[] dataItemsMap = new int[DSSettings.DataItems.Count * m_MapLength];

    //        for (int i = 0; i < DSSettings.ChangeItems.Count; i++)
    //        {
    //            dataItemsMap[i * m_MapLength + 0] = (int)DSSettings.DataItemsCalculationMap[DSSettings.ChangeItems[i].DataItem].FromData;
    //            dataItemsMap[i * m_MapLength + 0] = (int)DSSettings.DataItemsCalculationMap[DSSettings.ChangeItems[i].DataItem].OfData;
    //            dataItemsMap[i * m_MapLength + 0] = DSSettings.DataItemsCalculationMap[DSSettings.ChangeItems[i].DataItem].FromOffset;
    //            dataItemsMap[i * m_MapLength + 0] = DSSettings.DataItemsCalculationMap[DSSettings.ChangeItems[i].DataItem].OfOffset;
    //            dataItemsMap[i * m_MapLength + 0] = DSSettings.DataItemsCalculationMap[DSSettings.ChangeItems[i].DataItem].IsPositiveChange ? 1 : 0;
    //        }

    //        return dataItemsMap;
    //    }

    //    #endregion
    //}

}
