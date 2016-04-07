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

namespace StocksData
{
    internal class GPUPredictions : ILGPUModule
    {
        #region Members

        private readonly int m_DataSetWidth = (int)DataSet.DataColumns.NumOfColumns;

        private readonly int m_DataSetNumOfRows;

        private readonly int m_ChangesNum = DSSettings.ChangeItems.Count;

        private readonly int m_PredictionsNum = DSSettings.PredictionItems.Count;

        private readonly float m_ErrorRange = DSSettings.PredictionErrorRange;

        private DeviceMemory<int> m_DataItemsMapDM;
        private DeviceMemory<float> m_DataSetDM;
        private DeviceMemory<int> m_ChangesDataItemsDM;
        private DeviceMemory<byte> m_ChangesRangesDM;
        private DeviceMemory<int> m_PredictionsDataItemsDM;
        private DeviceMemory<byte> m_PredictionsRangesDM;
        private DeviceMemory<byte> m_PredictedChangesDM;
        private DeviceMemory<byte> m_ActualChangesDM;
        private DeviceMemory<float> m_PredictionResultsDM;

        #endregion

        #region Constructors

        public GPUPredictions(float[] dataset, int[] changesDataItems, byte[] changesRanges, int[] predictionDataItems, byte[] predictionRanges, int combinationsNum)
            : base(GPUModuleTarget.DefaultWorker)
        {
            
            m_DataSetNumOfRows= dataset.Length / m_DataSetWidth;

            // Allocate memory for initialization
            m_DataItemsMapDM = GPUWorker.Malloc(GetDataItemsMap());
            m_DataSetDM = GPUWorker.Malloc(dataset);
            m_ChangesDataItemsDM = GPUWorker.Malloc(changesDataItems);
            m_ChangesRangesDM = GPUWorker.Malloc(changesRanges);
            m_PredictionsDataItemsDM = GPUWorker.Malloc(predictionDataItems);
            m_PredictionsRangesDM = GPUWorker.Malloc(predictionRanges);
            m_PredictedChangesDM = GPUWorker.Malloc(new byte[m_DataSetNumOfRows * m_ChangesNum]);
            m_ActualChangesDM = GPUWorker.Malloc(new byte[m_DataSetNumOfRows * m_PredictionsNum]);


            // Initialize
            var block = new dim3(1024);
            var grid = new dim3(m_DataSetNumOfRows / block.x + 1);
            var lp = new LaunchParam(grid, block);

            GPULaunch(BuildPredictedChanges, lp, m_DataSetDM.Ptr, m_PredictedChangesDM.Ptr, m_DataItemsMapDM.Ptr, m_ChangesDataItemsDM.Ptr, m_ChangesRangesDM.Ptr); 
            GPULaunch(BuildActualChanges, lp, m_DataSetDM.Ptr, m_ActualChangesDM.Ptr, m_DataItemsMapDM.Ptr, m_PredictionsDataItemsDM.Ptr, m_PredictionsRangesDM.Ptr);

            //Free no more needed memory
            m_DataItemsMapDM.Dispose();
            m_DataSetDM.Dispose();
            m_ChangesDataItemsDM.Dispose();
            m_ChangesRangesDM.Dispose();

            m_DataItemsMapDM = null;
            m_DataSetDM = null;
            m_ChangesDataItemsDM = null;
            m_ChangesRangesDM = null;

            //Allocate memory for predictions caclucaltions
            m_PredictionResultsDM = GPUWorker.Malloc<float>(combinationsNum * m_PredictionsNum);
        }

        #endregion

        #region Interface

        public void FreeGPU()
        {
            m_PredictionsDataItemsDM.Dispose();
            m_PredictionsRangesDM.Dispose();
            m_PredictedChangesDM.Dispose();
            m_ActualChangesDM.Dispose();
            m_PredictionResultsDM.Dispose();

            m_PredictionsDataItemsDM = null;
            m_PredictionsRangesDM = null;
            m_PredictedChangesDM = null;
            m_ActualChangesDM = null;
            m_PredictionResultsDM = null;
        }

        public float[] PredictCombinations(byte[] combinations, byte combinationSize, int combinationsNum, int minimumChangesForPrediction, float minimumRelevantPredictionResult)
        {
            using (var dCombinations = GPUWorker.Malloc(combinations))
            {
                int numOfThreadsInBlock = 1024;
                int blockY = 1;// m_PredictionsNum;
                int blockX = numOfThreadsInBlock;// / blockY;
                int gridX = combinationsNum / blockX + 1;
                var block = new dim3(blockX);//, blockY);
                var grid = new dim3(gridX);
                var lp = new LaunchParam(grid, block);
                GPULaunch(PredictCombinations, lp, dCombinations.Ptr, combinationSize, m_PredictionResultsDM.Ptr, m_PredictedChangesDM.Ptr,
                    m_ActualChangesDM.Ptr, m_PredictionsRangesDM.Ptr, combinationsNum, minimumChangesForPrediction, minimumRelevantPredictionResult);
                return m_PredictionResultsDM.Gather();
            }
        }

        public float[] PredictCombinationsTest(byte[] combinations, byte combinationSize, int combinationsNum, int minimumChangesForPrediction, int numOfRows)
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
                GPULaunch(PredictCombinationsTest, lp, dCombinations.Ptr, combinationSize, m_PredictionResultsDM.Ptr, m_PredictedChangesDM.Ptr, 
                    m_ActualChangesDM.Ptr, m_PredictionsRangesDM.Ptr, combinationsNum, minimumChangesForPrediction, numOfRows);
                return m_PredictionResultsDM.Gather();
            }
        }


        #endregion

        #region Private Methods

        [Kernel]
        public void BuildPredictedChanges(deviceptr<float> dataSet, deviceptr<byte> predictedChanges, deviceptr<int> dataItemsMap, deviceptr<int> predictionsDataItems, deviceptr<byte> predictionsRanges)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentPredictedChanges = predictedChanges.Ptr(dataRow * m_ChangesNum);

            for (int predictionNum = 0; predictionNum < m_ChangesNum; predictionNum++)
            {
                int columnFrom = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 0];
                int columnOf = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 1];
                int isDifFromPrevDate = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 2];
                int isBigger = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 3];
                int range = predictionsRanges[predictionNum];
                deviceptr<float> currentDataSet = dataSet.Ptr((dataRow + range) * m_DataSetWidth);

                currentPredictedChanges[predictionNum] = (dataRow >= m_DataSetNumOfRows - range * 3) ?
                        (byte)0 : currentPredictedChanges[predictionNum] = IsPrediction(currentDataSet, range, columnFrom, columnOf, isDifFromPrevDate, isBigger, -m_ErrorRange, m_ErrorRange);

            }
        }

        [Kernel]
        public void BuildActualChanges(deviceptr<float> dataSet, deviceptr<byte> actualChanges, deviceptr<int> dataItemsMap, deviceptr<int> predictionsDataItems, deviceptr<byte> predictionsRanges)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentActualChanges = actualChanges.Ptr(dataRow * m_PredictionsNum);
            deviceptr<float> currentDataSet = dataSet.Ptr(dataRow * m_DataSetWidth);

            for (int predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
            {
                int columnFrom = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 0];
                int columnOf = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 1];
                int isDifFromPrevDate = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 2];
                int isBigger = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 3];
                int range = predictionsRanges[predictionNum];

                currentActualChanges[predictionNum] = (dataRow >= m_DataSetNumOfRows - range * 3) ?
                    (byte)0 : IsPrediction(currentDataSet, range, columnFrom, columnOf, isDifFromPrevDate, isBigger, m_ErrorRange, -m_ErrorRange);
            }
        }

        [Kernel]
        public void PredictCombinations(deviceptr<byte> combinationItems, byte combinationSize, deviceptr<float> predictionResults, deviceptr<byte> predictedChanges,
            deviceptr<byte> actualChanges, deviceptr<byte> predictionRanges, int combinationsNum, int minimumChangesForPrediction, float minimumRelevantPredictionResult)
        {
            var combinationNum = blockIdx.x * blockDim.x + threadIdx.x;

            if (combinationNum < combinationsNum)
            {
                deviceptr<byte> threadCombinationItems = combinationItems.Ptr(combinationNum * combinationSize);
                var threadActualPredictedChangeSum = __local__.Array<short>(m_PredictionsNum);

                for (byte predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
                {
                    threadActualPredictedChangeSum[predictionNum] = 0;
                }

                short predictedChangesSum = 0;
                for (int rowNum = 0; rowNum < m_DataSetNumOfRows; rowNum++)
                {
                    short predictedChange = 1;
                    for (byte itemNum = 0; itemNum < combinationSize; itemNum++)
                    {
                        predictedChange *= predictedChanges[rowNum * m_ChangesNum + threadCombinationItems[itemNum]];
                    }

                    predictedChangesSum += predictedChange;
                    for (byte predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
                    {
                        threadActualPredictedChangeSum[predictionNum] += (short)(predictedChange * actualChanges[rowNum * m_PredictionsNum + predictionNum]);
                    }
                }

                for (byte predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
                {
                    bool isRelevant = (predictedChangesSum > minimumChangesForPrediction)
                        && (float)threadActualPredictedChangeSum[predictionNum] / minimumChangesForPrediction >= minimumRelevantPredictionResult;
                    float predictionResult = isRelevant ? (float)threadActualPredictedChangeSum[predictionNum] / (float)predictedChangesSum : 0.0F;
                    predictionResults[combinationNum * m_PredictionsNum + predictionNum] = predictionResult;
                }
            }
        }

        [Kernel]
        public void PredictCombinationsTest(deviceptr<byte> combinationItems, byte combinationSize, deviceptr<float> predictionResults, deviceptr<byte> predictedChanges, 
            deviceptr<byte> actualChanges, deviceptr<byte> predictionRanges, int combinationsNum, int minimumChangesForPrediction, int numOfRows)
        {
            var combinationNum = blockIdx.x * blockDim.x + threadIdx.x;
            var predictionNum = threadIdx.y;

            if (combinationNum < combinationsNum)
            {
                deviceptr<byte> threadCombinationItems = combinationItems.Ptr(combinationNum * combinationSize);

                float predictedChangesSum = 0;
                float actualChangesSum = 0;
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
                float predictionResult = (predictedChangesSum > minimumChangesForPrediction) ? actualChangesSum / predictedChangesSum : 0.0F;
                predictionResults[combinationNum * m_PredictionsNum + predictionNum] = predictionResult;
            }
        }

        private byte IsPrediction(deviceptr<float> dataSet, int range, int dataColumFrom, int dataColumOf, int isDifFromPrevDate,  int isBigger, float biggerErrorBorder, float smallerErrorBorder)
        {
            int ofRow = isDifFromPrevDate * range;
            float sumOf = 0;
            float sumFrom = 0;
            for (int i = 0; i < range; i++)
            {
                sumOf += dataSet[(ofRow) * m_DataSetWidth + dataColumOf];
                sumFrom += dataSet[dataColumFrom];
            }

            return (byte)((isBigger == 1) ?
                (((sumFrom - sumOf) / sumOf / range) > biggerErrorBorder) ? 1 : 0
                :
                (((sumFrom - sumOf) / sumOf / range) < smallerErrorBorder) ? 1 : 0);
        }

        private int[] GetDataItemsMap()
        {
            int mapLength = 4;
            int[] dataItemsMap = new int[DSSettings.DataItems.Count * mapLength];

            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.OpenChange) * mapLength + 0] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.OpenChange) * mapLength + 1] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.OpenChange) * mapLength + 2] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.OpenChange) * mapLength + 3] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.CloseChange) * mapLength + 0] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.CloseChange) * mapLength + 1] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.CloseChange) * mapLength + 2] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.CloseChange) * mapLength + 3] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.VolumeChange) * mapLength + 0] = (int)DataSet.DataColumns.Volume;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.VolumeChange) * mapLength + 1] = (int)DataSet.DataColumns.Volume;            
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.VolumeChange) * mapLength + 2] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.VolumeChange) * mapLength + 3] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.CloseOpenDif) * mapLength + 0] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.CloseOpenDif) * mapLength + 1] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.CloseOpenDif) * mapLength + 2] = 0;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.CloseOpenDif) * mapLength + 3] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.OpenPrevCloseDif) * mapLength + 0] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.OpenPrevCloseDif) * mapLength + 1] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.OpenPrevCloseDif) * mapLength + 2] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.OpenPrevCloseDif) * mapLength + 3] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeOpenChange) * mapLength + 0] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeOpenChange) * mapLength + 1] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeOpenChange) * mapLength + 2] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeOpenChange) * mapLength + 3] = 0;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeCloseChange) * mapLength + 0] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeCloseChange) * mapLength + 1] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeCloseChange) * mapLength + 2] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeCloseChange) * mapLength + 3] = 0;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeVolumeChange) * mapLength + 0] = (int)DataSet.DataColumns.Volume;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeVolumeChange) * mapLength + 1] = (int)DataSet.DataColumns.Volume;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeVolumeChange) * mapLength + 2] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeVolumeChange) * mapLength + 3] = 0;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeCloseOpenDif) * mapLength + 0] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeCloseOpenDif) * mapLength + 1] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeCloseOpenDif) * mapLength + 2] = 0;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeCloseOpenDif) * mapLength + 3] = 0;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeOpenPrevCloseDif) * mapLength + 0] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeOpenPrevCloseDif) * mapLength + 1] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeOpenPrevCloseDif) * mapLength + 2] = 1;
            dataItemsMap[DSSettings.DataItems.IndexOf(DataItem.NegativeOpenPrevCloseDif) * mapLength + 3] = 0;

            return dataItemsMap;
        }

        #endregion
    }
}
