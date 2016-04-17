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

        private readonly double m_ErrorRange = DSSettings.PredictionErrorRange;

        private DeviceMemory<int> m_DataItemsMapDM;
        private DeviceMemory<double> m_DataSetDM;
        private DeviceMemory<int> m_ChangesDataItemsDM;
        private DeviceMemory<byte> m_ChangesRangesDM;
        private DeviceMemory<int> m_PredictionsDataItemsDM;
        private DeviceMemory<byte> m_PredictionsRangesDM;
        private DeviceMemory<byte> m_ChangesDM;
        private DeviceMemory<byte> m_PredictionsDM;
        private DeviceMemory<double> m_PredictionResultsDM;

        #endregion

        #region Constructors

        public GPUPredictions(double[] dataset, int[] changesDataItems, byte[] changesRanges, int[] predictionDataItems, byte[] predictionRanges, int combinationsNum)
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
            m_ChangesDM = GPUWorker.Malloc(new byte[m_DataSetNumOfRows * m_ChangesNum]);
            m_PredictionsDM = GPUWorker.Malloc(new byte[m_DataSetNumOfRows * m_PredictionsNum]);


            // Initialize
            var block = new dim3(1024);
            var grid = new dim3(m_DataSetNumOfRows / block.x + 1);
            var lp = new LaunchParam(grid, block);

            GPULaunch(BuildChanges, lp, m_DataSetDM.Ptr, m_ChangesDM.Ptr, m_DataItemsMapDM.Ptr, m_ChangesDataItemsDM.Ptr, m_ChangesRangesDM.Ptr); 
            GPULaunch(BuildPredictions, lp, m_DataSetDM.Ptr, m_PredictionsDM.Ptr, m_DataItemsMapDM.Ptr, m_PredictionsDataItemsDM.Ptr, m_PredictionsRangesDM.Ptr);

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
            m_PredictionResultsDM = GPUWorker.Malloc<double>(combinationsNum * m_PredictionsNum);
        }

        #endregion

        #region Interface

        public void FreeGPU()
        {
            m_PredictionsDataItemsDM.Dispose();
            m_PredictionsRangesDM.Dispose();
            m_ChangesDM.Dispose();
            m_PredictionsDM.Dispose();
            m_PredictionResultsDM.Dispose();

            m_PredictionsDataItemsDM = null;
            m_PredictionsRangesDM = null;
            m_ChangesDM = null;
            m_PredictionsDM = null;
            m_PredictionResultsDM = null;
        }

        public double[] PredictCombinations(byte[] combinations, byte combinationSize, int combinationsNum, int minimumChangesForPrediction, double minimumRelevantPredictionResult)
        {
            //using (var dPredictionsSum = GPUWorker.Malloc<short>(combinationsNum * m_PredictionsNum))
            using (var dCombinations = GPUWorker.Malloc(combinations))
            {
                int numOfThreadsInBlock = 1024;
                int blockY = 1;// m_PredictionsNum;
                int blockX = numOfThreadsInBlock;// / blockY;
                int gridX = combinationsNum / blockX + 1;
                var block = new dim3(blockX);//, blockY);
                var grid = new dim3(gridX);
                var lp = new LaunchParam(grid, block);
                GPULaunch(PredictCombinations, lp, dCombinations.Ptr, combinationSize, m_PredictionResultsDM.Ptr, m_ChangesDM.Ptr, m_PredictionsDM.Ptr,
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
        public void BuildChanges(deviceptr<double> dataSet, deviceptr<byte> changes, deviceptr<int> dataItemsMap, deviceptr<int> changesDataItems, deviceptr<byte> changesRanges)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentChanges = changes.Ptr(dataRow * m_ChangesNum);

            for (int changeNum = 0; changeNum < m_ChangesNum; changeNum++)
            {
                int columnFrom = dataItemsMap[changesDataItems[changeNum] * 4 + 0];
                int columnOf = dataItemsMap[changesDataItems[changeNum] * 4 + 1];
                int isDifFromPrevDate = dataItemsMap[changesDataItems[changeNum] * 4 + 2];
                int isBigger = dataItemsMap[changesDataItems[changeNum] * 4 + 3];
                int range = changesRanges[changeNum];
                deviceptr<double> currentDataSet = dataSet.Ptr((dataRow + range) * m_DataSetWidth);

                currentChanges[changeNum] = (dataRow >= m_DataSetNumOfRows - range * 3) ?
                        (byte)0 : currentChanges[changeNum] = IsPrediction(currentDataSet, range, columnFrom, columnOf, isDifFromPrevDate, isBigger, m_ErrorRange, -m_ErrorRange);

            }
        }

        [Kernel]
        public void BuildPredictions(deviceptr<double> dataSet, deviceptr<byte> predictions, deviceptr<int> dataItemsMap, deviceptr<int> predictionsDataItems, deviceptr<byte> predictionsRanges)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentPredictions = predictions.Ptr(dataRow * m_PredictionsNum);
            deviceptr<double> currentDataSet = dataSet.Ptr(dataRow * m_DataSetWidth);

            for (int predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
            {
                int columnFrom = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 0];
                int columnOf = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 1];
                int isDifFromPrevDate = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 2];
                int isBigger = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 3];
                int range = predictionsRanges[predictionNum];

                currentPredictions[predictionNum] = (dataRow >= m_DataSetNumOfRows - range * 3) ?
                    (byte)0 : IsPrediction(currentDataSet, range, columnFrom, columnOf, isDifFromPrevDate, isBigger, m_ErrorRange, -m_ErrorRange);
            }
        }

        [Kernel]
        public void PredictCombinations(deviceptr<byte> combinationItems, byte combinationSize, deviceptr<double> predictionResults, deviceptr<byte> changes, deviceptr<byte> predictions, 
            deviceptr<byte> predictionRanges, int combinationsNum, int minimumChangesForPrediction, double minimumRelevantPredictionResult)//, deviceptr<short> predictionsSum)
        {
            var combinationNum = blockIdx.x * blockDim.x + threadIdx.x;

            if (combinationNum < combinationsNum)
            {
                deviceptr<byte> threadCombinationItems = combinationItems.Ptr(combinationNum * combinationSize);
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
                        change *= changes[rowNum * m_ChangesNum + threadCombinationItems[itemNum]];
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

        private byte IsPrediction(deviceptr<double> dataSet, int range, int dataColumFrom, int dataColumOf, int isDifFromPrevDate,  int isBigger, double biggerErrorBorder, double smallerErrorBorder)
        {
            int ofRow = isDifFromPrevDate * range;
            double sumOf = 0;
            double sumFrom = 0;
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


    internal class GPULatestPredictions : ILGPUModule
    {
        #region Members

        private readonly int m_DataSetWidth = (int)DataSet.DataColumns.NumOfColumns;

        private readonly int m_DataSetNumOfRows;

        private readonly int m_ChangesNum = DSSettings.ChangeItems.Count;

        private readonly int m_PredictionsNum = DSSettings.PredictionItems.Count;

        private readonly double m_ErrorRange = DSSettings.PredictionErrorRange;

        private DeviceMemory<int> m_DataItemsMapDM;
        private DeviceMemory<double> m_DataSetDM;
        private DeviceMemory<int> m_ChangesDataItemsDM;
        private DeviceMemory<byte> m_ChangesRangesDM;
        private DeviceMemory<int> m_PredictionsDataItemsDM;
        private DeviceMemory<byte> m_PredictionsRangesDM;
        private DeviceMemory<byte> m_ChangesDM;
        private DeviceMemory<byte> m_PredictionsDM;
        private DeviceMemory<double> m_PredictionResultsDM;

        #endregion

        #region Constructors

        public GPULatestPredictions(double[] datasetsData, int numOfDataSets, int[] changesDataItems, byte[] changesRanges, int[] predictionDataItems, byte[] predictionRanges, int combinationsNum)
            : base(GPUModuleTarget.DefaultWorker)
        {

            m_DataSetNumOfRows = DSSettings.PredictionsSize * numOfDataSets;

            // Allocate memory for initialization
            m_DataItemsMapDM = GPUWorker.Malloc(GetDataItemsMap());
            m_DataSetDM = GPUWorker.Malloc(datasetsData);
            m_ChangesDataItemsDM = GPUWorker.Malloc(changesDataItems);
            m_ChangesRangesDM = GPUWorker.Malloc(changesRanges);
            m_PredictionsDataItemsDM = GPUWorker.Malloc(predictionDataItems);
            m_PredictionsRangesDM = GPUWorker.Malloc(predictionRanges);
            m_ChangesDM = GPUWorker.Malloc(new byte[numOfDataSets * DSSettings.DataSetForPredictionsSize * m_ChangesNum]);
            m_PredictionsDM = GPUWorker.Malloc(new byte[numOfDataSets * DSSettings.DataSetForPredictionsSize * m_PredictionsNum]);


            // Initialize
            var block = new dim3(DSSettings.PredictionsSize);
            var grid = new dim3(1);
            var lp = new LaunchParam(grid, block);

            for (int dataSetNum = 0; dataSetNum < numOfDataSets; dataSetNum++)
            {
                GPULaunch(BuildChanges, lp, dataSetNum, DSSettings.DataSetForPredictionsSize, DSSettings.PredictionsSize, m_DataSetDM.Ptr, m_ChangesDM.Ptr, m_DataItemsMapDM.Ptr, m_ChangesDataItemsDM.Ptr, m_ChangesRangesDM.Ptr);
                GPULaunch(BuildPredictions, lp, dataSetNum, DSSettings.DataSetForPredictionsSize, DSSettings.PredictionsSize, m_DataSetDM.Ptr, m_PredictionsDM.Ptr, m_DataItemsMapDM.Ptr, m_PredictionsDataItemsDM.Ptr, m_PredictionsRangesDM.Ptr);
            }

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
            m_PredictionResultsDM = GPUWorker.Malloc<double>(combinationsNum * m_PredictionsNum);
        }

        #endregion

        #region Interface

        public void FreeGPU()
        {
            m_PredictionsDataItemsDM.Dispose();
            m_PredictionsRangesDM.Dispose();
            m_ChangesDM.Dispose();
            m_PredictionsDM.Dispose();
            m_PredictionResultsDM.Dispose();

            m_PredictionsDataItemsDM = null;
            m_PredictionsRangesDM = null;
            m_ChangesDM = null;
            m_PredictionsDM = null;
            m_PredictionResultsDM = null;
        }

        public double[] PredictCombinations(byte[] combinations, byte combinationSize, int combinationsNum, int minimumChangesForPrediction, double minimumRelevantPredictionResult)
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
                GPULaunch(PredictCombinations, lp, dCombinations.Ptr, combinationSize, m_PredictionResultsDM.Ptr, m_ChangesDM.Ptr,
                    m_PredictionsDM.Ptr, m_PredictionsRangesDM.Ptr, combinationsNum, minimumChangesForPrediction, minimumRelevantPredictionResult);
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
        public void BuildChanges(int dataSetNum, int dataSetSize, int dataSetChangesSize, deviceptr<double> dataSet, deviceptr<byte> changes, deviceptr<int> dataItemsMap, 
            deviceptr<int> changesDataItems, deviceptr<byte> changesRanges)
        {
            var dataRow =blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentChanges = changes.Ptr((dataSetNum * dataSetChangesSize + dataRow) * m_ChangesNum);

            for (int changeNum = 0; changeNum < m_ChangesNum; changeNum++)
            {
                int columnFrom = dataItemsMap[changesDataItems[changeNum] * 4 + 0];
                int columnOf = dataItemsMap[changesDataItems[changeNum] * 4 + 1];
                int isDifFromPrevDate = dataItemsMap[changesDataItems[changeNum] * 4 + 2];
                int isBigger = dataItemsMap[changesDataItems[changeNum] * 4 + 3];
                int range = changesRanges[changeNum];
                deviceptr<double> currentDataSet = dataSet.Ptr((dataSetNum * dataSetSize + dataRow + range) * m_DataSetWidth);

                currentChanges[changeNum] = IsChange(currentDataSet, range, columnFrom, columnOf, isDifFromPrevDate, isBigger, m_ErrorRange, -m_ErrorRange);

            }
        }

        [Kernel]
        public void BuildPredictions(int dataSetNum, int dataSetSize, int dataSetPredictionsSize, deviceptr<double> dataSet, deviceptr<byte> predictions, deviceptr<int> dataItemsMap, deviceptr<int> predictionsDataItems, deviceptr<byte> predictionsRanges)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentPredictions = predictions.Ptr((dataSetNum * dataSetPredictionsSize + dataRow) * m_PredictionsNum);
            deviceptr<double> currentDataSet = dataSet.Ptr((dataSetNum * dataSetSize + dataRow) * m_DataSetWidth);

            for (int predictionNum = 0; predictionNum < m_PredictionsNum; predictionNum++)
            {
                int columnFrom = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 0];
                int columnOf = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 1];
                int isDifFromPrevDate = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 2];
                int isBigger = dataItemsMap[predictionsDataItems[predictionNum] * 4 + 3];
                int range = predictionsRanges[predictionNum];

                currentPredictions[predictionNum] = IsChange(currentDataSet, range, columnFrom, columnOf, isDifFromPrevDate, isBigger, m_ErrorRange, -m_ErrorRange);
            }
        }

        [Kernel]
        public void PredictCombinations(deviceptr<byte> combinationItems, byte combinationSize, deviceptr<double> predictionResults, deviceptr<byte> predictedChanges,
            deviceptr<byte> actualChanges, deviceptr<byte> predictionRanges, int combinationsNum, int minimumChangesForPrediction, double minimumRelevantPredictionResult)
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
                        && (double)threadActualPredictedChangeSum[predictionNum] / minimumChangesForPrediction >= minimumRelevantPredictionResult;
                    double predictionResult = isRelevant ? (double)threadActualPredictedChangeSum[predictionNum] / (double)predictedChangesSum : 0.0;
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

        private byte IsChange(deviceptr<double> dataSet, int range, int dataColumFrom, int dataColumOf, int isDifFromPrevDate, int isBigger, double biggerErrorBorder, double smallerErrorBorder)
        {
            int ofRow = isDifFromPrevDate * range;
            double sumOf = 0;
            double sumFrom = 0;
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
