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
    internal class GPUAnalyzer : ILGPUModule
    {
        #region Members

        private readonly int m_DataSetWidth = (int)DataSet.DataColumns.NumOfColumns;

        private readonly int m_DataSetNumOfRows;

        private readonly int m_PredictionsNum = DSSettings.PredictionItems.Count;

        private readonly int m_AnalyzesNum = DSSettings.AnalyzeItems.Count;

        private readonly float m_ErrorRange = DSSettings.PredictionErrorRange;

        private DeviceMemory<int> m_DataItemsMapDM;
        private DeviceMemory<float> m_DataSetDM;
        private DeviceMemory<int> m_PredictionsDataItemsDM;
        private DeviceMemory<byte> m_PredictionsRangesDM;
        private DeviceMemory<int> m_AnalyzesDataItemsDM;
        private DeviceMemory<byte> m_AnalyzesRangesDM;
        private DeviceMemory<byte> m_PredictedChangesDM;
        private DeviceMemory<byte> m_ActualChangesDM;
        private DeviceMemory<float> m_AnalyzeResultsDM;

        #endregion

        #region Constructors

        public GPUAnalyzer(float[] dataset, int[] predictionsDataItems, byte[] predictionsRanges, int[] analyzesDataItems, byte[] analyzesRanges, int combinationsNum)
            : base(GPUModuleTarget.DefaultWorker)
        {
            
            m_DataSetNumOfRows= dataset.Length / m_DataSetWidth;

            // Allocate memory for initialization
            m_DataItemsMapDM = GPUWorker.Malloc(GetDataItemsMap());
            m_DataSetDM = GPUWorker.Malloc(dataset);
            m_PredictionsDataItemsDM = GPUWorker.Malloc(predictionsDataItems);
            m_PredictionsRangesDM = GPUWorker.Malloc(predictionsRanges);
            m_AnalyzesDataItemsDM = GPUWorker.Malloc(analyzesDataItems);
            m_AnalyzesRangesDM = GPUWorker.Malloc(analyzesRanges);
            m_PredictedChangesDM = GPUWorker.Malloc(new byte[m_DataSetNumOfRows * m_PredictionsNum]);
            m_ActualChangesDM = GPUWorker.Malloc(new byte[m_DataSetNumOfRows * m_AnalyzesNum]);


            // Initialize
            var block = new dim3(1024);
            var grid = new dim3(m_DataSetNumOfRows / block.x + 1);
            var lp = new LaunchParam(grid, block);

            GPULaunch(BuildPredictedChanges, lp, m_DataSetDM.Ptr, m_PredictedChangesDM.Ptr, m_DataItemsMapDM.Ptr, m_PredictionsDataItemsDM.Ptr, m_PredictionsRangesDM.Ptr); 
            GPULaunch(BuildActualChanges, lp, m_DataSetDM.Ptr, m_ActualChangesDM.Ptr, m_DataItemsMapDM.Ptr, m_AnalyzesDataItemsDM.Ptr, m_AnalyzesRangesDM.Ptr);

            //Free no more needed memory
            m_DataItemsMapDM.Dispose();
            m_DataSetDM.Dispose();
            m_PredictionsDataItemsDM.Dispose();
            m_PredictionsRangesDM.Dispose();

            m_DataItemsMapDM = null;
            m_DataSetDM = null;
            m_PredictionsDataItemsDM = null;
            m_PredictionsRangesDM = null;

            //Allocate memory for analyzer caclucaltions
            m_AnalyzeResultsDM = GPUWorker.Malloc<float>(combinationsNum * m_AnalyzesNum);
        }

        #endregion

        #region Interface

        public void FreeGPU()
        {
            m_AnalyzesDataItemsDM.Dispose();
            m_AnalyzesRangesDM.Dispose();
            m_PredictedChangesDM.Dispose();
            m_ActualChangesDM.Dispose();
            m_AnalyzeResultsDM.Dispose();

            m_AnalyzesDataItemsDM = null;
            m_AnalyzesRangesDM = null;
            m_PredictedChangesDM = null;
            m_ActualChangesDM = null;
            m_AnalyzeResultsDM = null;
        }

        public float[] AnalyzeCombinations(byte[] combinations, byte combinationSize, int combinationsNum, int minimumPredictionsForAnalyze, float minimumRelevantAnalyzeResult)
        {
            using (var dCombinations = GPUWorker.Malloc(combinations))
            {
                int numOfThreadsInBlock = 1024;
                int blockY = 1;// m_AnalyzesNum;
                int blockX = numOfThreadsInBlock;// / blockY;
                int gridX = combinationsNum / blockX + 1;
                var block = new dim3(blockX);//, blockY);
                var grid = new dim3(gridX);
                var lp = new LaunchParam(grid, block);
                GPULaunch(AnalyzeCombinations, lp, dCombinations.Ptr, combinationSize, m_AnalyzeResultsDM.Ptr, m_PredictedChangesDM.Ptr,
                    m_ActualChangesDM.Ptr, m_AnalyzesRangesDM.Ptr, combinationsNum, minimumPredictionsForAnalyze, minimumRelevantAnalyzeResult);
                return m_AnalyzeResultsDM.Gather();
            }
        }

        public float[] AnalyzeCombinationsTest(byte[] combinations, byte combinationSize, int combinationsNum, int minimumPredictionsForAnalyze, int numOfRows)
        {
            using (var dCombinations = GPUWorker.Malloc(combinations))
            {
                int numOfThreadsInBlock = 1024;
                int blockY = m_AnalyzesNum;
                int blockX = numOfThreadsInBlock / blockY;
                int gridX = combinationsNum / blockX + 1;
                var block = new dim3(blockX, blockY);
                var grid = new dim3(gridX);
                var lp = new LaunchParam(grid, block);
                GPULaunch(AnalyzeCombinationsTest, lp, dCombinations.Ptr, combinationSize, m_AnalyzeResultsDM.Ptr, m_PredictedChangesDM.Ptr, 
                    m_ActualChangesDM.Ptr, m_AnalyzesRangesDM.Ptr, combinationsNum, minimumPredictionsForAnalyze, numOfRows);
                return m_AnalyzeResultsDM.Gather();
            }
        }


        #endregion

        #region Private Methods

        [Kernel]
        public void BuildPredictedChanges(deviceptr<float> dataSet, deviceptr<byte> predictedChanges, deviceptr<int> dataItemsMap, deviceptr<int> predictionsDataItems, deviceptr<byte> predictionsRanges)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentPredictedChanges = predictedChanges.Ptr(dataRow * m_PredictionsNum);

            for (int predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
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
        public void BuildActualChanges(deviceptr<float> dataSet, deviceptr<byte> actualChanges, deviceptr<int> dataItemsMap, deviceptr<int> analyzesDataItems, deviceptr<byte> analyzesRanges)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentActualChanges = actualChanges.Ptr(dataRow * m_AnalyzesNum);
            deviceptr<float> currentDataSet = dataSet.Ptr(dataRow * m_DataSetWidth);

            for (int analyzeNum = 0; analyzeNum < m_AnalyzesNum; analyzeNum++)
            {
                int columnFrom = dataItemsMap[analyzesDataItems[analyzeNum] * 4 + 0];
                int columnOf = dataItemsMap[analyzesDataItems[analyzeNum] * 4 + 1];
                int isDifFromPrevDate = dataItemsMap[analyzesDataItems[analyzeNum] * 4 + 2];
                int isBigger = dataItemsMap[analyzesDataItems[analyzeNum] * 4 + 3];
                int range = analyzesRanges[analyzeNum];

                currentActualChanges[analyzeNum] = (dataRow >= m_DataSetNumOfRows - range * 3) ?
                    (byte)0 : IsPrediction(currentDataSet, range, columnFrom, columnOf, isDifFromPrevDate, isBigger, m_ErrorRange, -m_ErrorRange);
            }
        }

        [Kernel]
        public void AnalyzeCombinations(deviceptr<byte> combinationItems, byte combinationSize, deviceptr<float> analyzeResults, deviceptr<byte> predictedChanges,
            deviceptr<byte> actualChanges, deviceptr<byte> analyzeRanges, int combinationsNum, int minimumPredictionsForAnalyze, float minimumRelevantAnalyzeResult)
        {
            var combinationNum = blockIdx.x * blockDim.x + threadIdx.x;

            if (combinationNum < combinationsNum)
            {
                deviceptr<byte> threadCombinationItems = combinationItems.Ptr(combinationNum * combinationSize);
                var threadActualPredictedChangeSum = __local__.Array<short>(m_AnalyzesNum);

                for (byte analyzeNum = 0; analyzeNum < m_AnalyzesNum; analyzeNum++)
                {
                    threadActualPredictedChangeSum[analyzeNum] = 0;
                }

                short predictedChangesSum = 0;
                for (int rowNum = 0; rowNum < m_DataSetNumOfRows; rowNum++)
                {
                    short predictedChange = 1;
                    for (byte itemNum = 0; itemNum < combinationSize; itemNum++)
                    {
                        predictedChange *= predictedChanges[rowNum * m_PredictionsNum + threadCombinationItems[itemNum]];
                    }

                    predictedChangesSum += predictedChange;
                    for (byte analyzeNum = 0; analyzeNum < m_AnalyzesNum; analyzeNum++)
                    {
                        threadActualPredictedChangeSum[analyzeNum] += (short)(predictedChange * actualChanges[rowNum * m_AnalyzesNum + analyzeNum]);
                    }
                }

                for (byte analyzeNum = 0; analyzeNum < m_AnalyzesNum; analyzeNum++)
                {
                    bool isRelevant = (predictedChangesSum > minimumPredictionsForAnalyze)
                        && (float)threadActualPredictedChangeSum[analyzeNum] / minimumPredictionsForAnalyze >= minimumRelevantAnalyzeResult;
                    float analyzeResult = isRelevant ? (float)threadActualPredictedChangeSum[analyzeNum] / (float)predictedChangesSum : 0.0F;
                    analyzeResults[combinationNum * m_AnalyzesNum + analyzeNum] = analyzeResult;
                }
            }
        }

        [Kernel]
        public void AnalyzeCombinationsTest(deviceptr<byte> combinationItems, byte combinationSize, deviceptr<float> analyzeResults, deviceptr<byte> predictedChanges, 
            deviceptr<byte> actualChanges, deviceptr<byte> analyzeRanges, int combinationsNum, int minimumPredictionsForAnalyze, int numOfRows)
        {
            var combinationNum = blockIdx.x * blockDim.x + threadIdx.x;
            var analyzeNum = threadIdx.y;

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
                        predictedChange *= predictedChanges[rowNum * m_PredictionsNum + threadCombinationItems[itemNum]];
                    }

                    predictedChangesSum += predictedChange;
                    actualChangesSum += predictedChange * actualChanges[rowNum * m_AnalyzesNum + analyzeNum];
                }
                float analyzeResult = (predictedChangesSum > minimumPredictionsForAnalyze) ? actualChangesSum / predictedChangesSum : 0.0F;
                analyzeResults[combinationNum * m_AnalyzesNum + analyzeNum] = analyzeResult;
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
