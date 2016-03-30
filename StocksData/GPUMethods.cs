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
    internal class GPUChanges : ILGPUModule
    {
        #region Members

        private readonly Func<double, double, double> _funcOp;

        #endregion

        #region Constructors

        public GPUChanges(Func<double, double, double> funcOp1) : base(GPUModuleTarget.DefaultWorker)
        {
            _funcOp = funcOp1;
        }

        #endregion

        #region Interface

        public static double[] CalculateChangesInPercents(double[] data, int height)
        {
            Func<double, double, double> changeOp = (x, y) => (x - y) / y;
            using (var loadedModule = new GPUChanges(changeOp))
            {
                return loadedModule.PercentageChangeMap(data, (int)DataSet.DataColumns.NumOfColumns, (int)ChangesDataSet.DataColumns.NumOfColumns, height);
            }
        }

        public double[] PercentageChangeMap(double[] inputs, int inputWidth, int outputWidth, int height)
        {
            using (var dInputs = GPUWorker.Malloc(inputs))
            using (var dOutputs = GPUWorker.Malloc<double>(outputWidth * (height - 1)))
            {
                var block = new dim3(512);
                var grid = new dim3(height / block.x + 1);
                var lp = new LaunchParam(grid, block);
                GPULaunch(PercentageChangeKernel, lp, dOutputs.Ptr, dInputs.Ptr, inputWidth, outputWidth, height);
                return dOutputs.Gather();
            }
        }

        [Kernel]
        public void PercentageChangeKernel(deviceptr<double> outputs, deviceptr<double> inputs, int inputWidth, int outputWidth, int height)
        {
            var inputStart = (blockIdx.x * blockDim.x + threadIdx.x) * inputWidth;
            var inputStride = gridDim.x * blockDim.x * inputWidth;

            var outputStart = (blockIdx.x * blockDim.x + threadIdx.x) * outputWidth;
            var outputStride = gridDim.x * blockDim.x * outputWidth;

            var outIndex = outputStart;
            for (var inIndex = inputStart; inIndex < height * inputWidth - 1; inIndex += inputStride)
            {
                outputs[outIndex + (int)ChangesDataSet.DataColumns.Date] = inputs[inIndex + (int)DataSet.DataColumns.Date];
                outputs[outIndex + (int)ChangesDataSet.DataColumns.OpenChange] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.Open], inputs[inIndex + (int)DataSet.DataColumns.Open + inputWidth]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.HighChange] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.High], inputs[inIndex + (int)DataSet.DataColumns.High + inputWidth]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.LowChange] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.Low], inputs[inIndex + (int)DataSet.DataColumns.Low + inputWidth]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.CloseChange] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.Close], inputs[inIndex + (int)DataSet.DataColumns.Close + inputWidth]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.VolumeChange] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.Volume], inputs[inIndex + (int)DataSet.DataColumns.Volume + inputWidth]); ;
                outputs[outIndex + (int)ChangesDataSet.DataColumns.HighLowDif] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.High], inputs[inIndex + (int)DataSet.DataColumns.Low]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.HighOpenDif] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.High], inputs[inIndex + (int)DataSet.DataColumns.Open]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.LowOpenDif] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.Low], inputs[inIndex + (int)DataSet.DataColumns.Open]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.CloseOpenDif] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.Close], inputs[inIndex + (int)DataSet.DataColumns.Open]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.HighPrevCloseDif] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.High], inputs[inIndex + (int)DataSet.DataColumns.Close + inputWidth]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.LowPrevCloseDif] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.Low], inputs[inIndex + (int)DataSet.DataColumns.Close + inputWidth]);
                outputs[outIndex + (int)ChangesDataSet.DataColumns.OpenPrevCloseDif] = _funcOp(inputs[inIndex + (int)DataSet.DataColumns.Open], inputs[inIndex + (int)DataSet.DataColumns.Close + inputWidth]);

                outIndex += outputStride;
            }
        }

        #endregion
    }
    internal class GPUPredictions : ILGPUModule
    {
        #region Constructors

        public GPUPredictions() : base(GPUModuleTarget.DefaultWorker)
        {
        }

        #endregion

        #region Predictions Calculate

        public static double[] CalculatePredictions(double[] data, int height)
        {
            using (var loadedModule = new GPUPredictions())
            {
                return loadedModule.PredictionMap(data, (int)ChangesDataSet.DataColumns.NumOfColumns, (int)PredictionsDataSet.DataColumns.NumOfColumns, height);
            }
        }

        public double[] PredictionMap(double[] inputs, int inputWidth, int outputWidth, int height)
        {
            using (var dInputs = GPUWorker.Malloc(inputs))
            using (var dOutputs = GPUWorker.Malloc<double>(outputWidth * (height * Constants.DepthsNum - Constants.NumOfOutOfRangePredictions)))
            {
                var block = new dim3(1024);
                var grid = new dim3(height / block.x + 1);
                var lp = new LaunchParam(grid, block);
                GPULaunch(PredictionKernel, lp, dOutputs.Ptr, dInputs.Ptr, inputWidth, outputWidth, height);
                return dOutputs.Gather();
            }
        }

        [Kernel]
        public void PredictionKernel(deviceptr<double> outputs, deviceptr<double> inputs, int inputWidth, int outputWidth, int height)
        {
            var inputStart = blockIdx.x * blockDim.x + threadIdx.x;
            var inputStride = gridDim.x * blockDim.x;


            // inIndex the rowNumber
            for (var inStartRow = inputStart; inStartRow < (height - Constants.MinDepthRange); inStartRow += inputStride)
            {
                var outStartRow = inStartRow;
                for (int depth = Constants.MinDepthRange; depth <= Constants.MaxDepthRange; depth++)
                {
                    if ((inStartRow + depth) >= height)
                    {
                        continue;
                    }

                    outputs[outStartRow * outputWidth + (int)PredictionsDataSet.DataColumns.Depth] = depth;
                    outputs[outStartRow * outputWidth + (int)PredictionsDataSet.DataColumns.Date] = inputs[inStartRow * inputWidth + (int)DataSet.DataColumns.Date];

                    int outColumnCell = outStartRow * outputWidth + (int)PredictionsDataSet.DataColumns.OpenChange;
                    for (int inColumnIndex = (int)ChangesDataSet.DataColumns.OpenChange; inColumnIndex < (int)ChangesDataSet.DataColumns.NumOfColumns; inColumnIndex++)
                    {
                        double sum = 0;
                        for (var inCurrentCell = (inStartRow + 1) * inputWidth + inColumnIndex; inCurrentCell < (inStartRow + depth + 1) * inputWidth; inCurrentCell += inputWidth)
                        {
                            sum += inputs[inCurrentCell];
                        }

                        outputs[outColumnCell] = sum;
                        outColumnCell++;
                    }

                    outStartRow += (height - depth);
                }
            }
        }

        #endregion
    }
    internal class GPUAnalyze : ILGPUModule
    {
        #region Members

        private readonly int m_ChangesWidth = (int)ChangesDataSet.DataColumns.NumOfColumns;

        private readonly int m_ChangesHeight;

        private readonly int m_PredictionsWidth = (int)PredictionsDataSet.DataColumns.NumOfColumns;

        private readonly int m_PredictionsHeight;

        private readonly int m_CombinationsNum;

        private readonly int m_NumOfAnalyzePredictions = Constants.AnalyzeChangesList.Count;

        private readonly int m_AnalyzeWidth = Constants.AnalyzeChangesList.Count * Constants.DepthsNum * 2;

        #endregion

        #region Constructors

        public GPUAnalyze(double[] changes, double[] predictions, AnalyzesDataSet.AnalyzeCombination[] combinations) : base(GPUModuleTarget.DefaultWorker)
        {

            m_ChangesHeight = changes.Length / m_ChangesWidth;

            m_PredictionsHeight = predictions.Length / m_PredictionsWidth;

            m_CombinationsNum = combinations.Length;
        }

        #endregion

        #region Interface

        public static double[] AnalyzePredictions(double[] changes,double[] predictions, AnalyzesDataSet.AnalyzeCombination[] combinations)
        {
            using (var loadedModule = new GPUAnalyze(changes, predictions, combinations))
            {
                return loadedModule.AnalyzeMap(changes, predictions, combinations);
            }
        }

        public double[] AnalyzeMap(double[] changes, double[] predictions, AnalyzesDataSet.AnalyzeCombination[] combinations)
        {
            using (var dChanges = GPUWorker.Malloc(changes))
            using (var dPredictions = GPUWorker.Malloc(predictions))
            using (var dCombinations = GPUWorker.Malloc(combinations))
            using (var dAnalyzeChanges = GPUWorker.Malloc(Constants.AnalyzeChangesList.ToArray()))
            using (var dAnalyzeResults = GPUWorker.Malloc<double>(m_AnalyzeWidth * combinations.Length))
            {
                var block = new dim3(512);
                var grid = new dim3(combinations.Length / block.x + 1);
                var lp = new LaunchParam(grid, block);
                GPULaunch(AnalyzeKernel, lp, dChanges.Ptr, dPredictions.Ptr, dCombinations.Ptr, dAnalyzeChanges.Ptr, dAnalyzeResults.Ptr);
                return dAnalyzeResults.Gather();
            }
        }

        [Kernel]
        public void AnalyzeKernel(deviceptr<double> changes, deviceptr<double> predictions, deviceptr<AnalyzesDataSet.AnalyzeCombination> combinations, deviceptr<ChangesDataSet.DataColumns> analyzeChanges, deviceptr<double> analyzeResults)
        {
            var combinationStart = blockIdx.x * blockDim.x + threadIdx.x;
            var combinationStride = gridDim.x * blockDim.x;

            for (var combination = combinationStart; combination < m_CombinationsNum; combination += combinationStride)
            {
                DoAnalyzeCombination(combinations[combination], changes, predictions, analyzeResults, combination * m_AnalyzeWidth);
            }
        }

        private void DoAnalyzeCombination(AnalyzesDataSet.AnalyzeCombination analyzeCombination, deviceptr<double> changes, deviceptr<double> predictions, deviceptr<double> analyzeResults, int analyzePosition)
        {
            int predictionRow = 0;
            while (predictionRow < m_PredictionsHeight)
            {
                predictionRow = AnalyzePredictionDepth(predictionRow, analyzeCombination, changes, predictions, analyzeResults, analyzePosition);
            }
        }

        private int AnalyzePredictionDepth(int predictionRow, AnalyzesDataSet.AnalyzeCombination analyzeCombination, deviceptr<double> changes, deviceptr<double> predictions, deviceptr<double> analyzeResults, int analyzePosition)
        {
            int localPredictionRow = predictionRow;
            int localAnalyzePosition = analyzePosition;
            var upCorrectPredictions = __local__.Array<double>(Constants.AnalyzeChangesList.Count);
            var downCorrectPredictions = __local__.Array<double>(Constants.AnalyzeChangesList.Count);
            var predictionsNum = __local__.Array<double>(Constants.AnalyzeChangesList.Count);

            var upPredictionsResults = __local__.Array<double>(Constants.AnalyzeChangesList.Count);
            var downPredictionsResults = __local__.Array<double>(Constants.AnalyzeChangesList.Count);

            var relevantHistoriesNum = __local__.Array<double>(Constants.AnalyzeChangesList.Count);

            double currentDepth = predictions[localPredictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns];
            int depthStartRow = localPredictionRow;
            //int changesRow = 0;

            for (int i = 0; i < Constants.AnalyzeChangesList.Count; i++)
            {
                upCorrectPredictions[i] = 0.0;
                downCorrectPredictions[i] = 0.0;
                predictionsNum[i] = 0.0;
            }

            //for (; localPredictionRow < m_PredictionsHeight && localPredictionRow < depthStartRow + Constants.RelevantHistory; localPredictionRow++)
            //{
            //    for (int predictedChangeNum = 0; predictedChangeNum < Constants.AnalyzeChangesList.Count; predictedChangeNum++)
            //    {
            //        double prediction = GetPredictionResult(predictions, localPredictionRow, analyzeCombination);
            //        double actualChange = changes[changesRow * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

            //        if (prediction > 0.0)
            //        {
            //            predictionsNum[predictedChangeNum]++;

            //            if (actualChange > Constants.PredictionErrorRange)
            //            {
            //                upCorrectPredictions[predictedChangeNum]++;
            //            }
            //            if (actualChange < -Constants.PredictionErrorRange)
            //            {
            //                downCorrectPredictions[predictedChangeNum]++;
            //            }
            //        }
            //    }

            //    changesRow++;
            //}

            //for (int i = 0; i < Constants.AnalyzeChangesList.Count; i++)
            //{
            //    if (predictionsNum[i] > 0)
            //    {
            //        upPredictionsResults[i] = upCorrectPredictions[i] / predictionsNum[i];
            //        downPredictionsResults[i] = downCorrectPredictions[i] / predictionsNum[i];
            //        relevantHistoriesNum[i] = 2;
            //    }
            //    else
            //    {
            //        relevantHistoriesNum[i] = 0;
            //    }
            //}

            //for (; localPredictionRow < m_PredictionsHeight && predictions[localPredictionRow * (int)PredictionsDataSet.DataColumns.NumOfColumns] == currentDepth; localPredictionRow++)
            //{
            //    for (int predictedChangeNum = 0; predictedChangeNum < Constants.AnalyzeChangesList.Count; predictedChangeNum++)
            //    {
            //        double prediction = GetPredictionResult(predictions, localPredictionRow - Constants.RelevantHistory, analyzeCombination);
            //        double actualChange = changes[(changesRow - Constants.RelevantHistory) * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

            //        if (prediction > 0.0)
            //        {
            //            predictionsNum[predictedChangeNum]--;

            //            if (actualChange > Constants.PredictionErrorRange)
            //            {
            //                upCorrectPredictions[predictedChangeNum]--;
            //            }
            //            if (actualChange < -Constants.PredictionErrorRange)
            //            {
            //                downCorrectPredictions[predictedChangeNum]--;
            //            }
            //        }

            //        prediction = GetPredictionResult(predictions, localPredictionRow, analyzeCombination);
            //        actualChange = changes[changesRow * (int)ChangesDataSet.DataColumns.NumOfColumns + (int)Constants.AnalyzeChangesList[predictedChangeNum]];

            //        if (prediction > 0.0)
            //        {
            //            predictionsNum[predictedChangeNum]++;

            //            if (actualChange > Constants.PredictionErrorRange)
            //            {
            //                upCorrectPredictions[predictedChangeNum]++;
            //            }
            //            if (actualChange < -Constants.PredictionErrorRange)
            //            {
            //                downCorrectPredictions[predictedChangeNum]++;
            //            }
            //        }

            //        if (predictionsNum[predictedChangeNum] > 0)
            //        {
            //            if (relevantHistoriesNum[predictedChangeNum] > 0)
            //            {
            //                upPredictionsResults[predictedChangeNum] += (upCorrectPredictions[predictedChangeNum] / predictionsNum[predictedChangeNum] - upPredictionsResults[predictedChangeNum]) / relevantHistoriesNum[predictedChangeNum];
            //                downPredictionsResults[predictedChangeNum] += (downCorrectPredictions[predictedChangeNum] / predictionsNum[predictedChangeNum] - downPredictionsResults[predictedChangeNum]) / relevantHistoriesNum[predictedChangeNum];
            //                relevantHistoriesNum[predictedChangeNum]++;
            //            }
            //            else
            //            {
            //                upPredictionsResults[predictedChangeNum] = upCorrectPredictions[predictedChangeNum] / predictionsNum[predictedChangeNum];
            //                downPredictionsResults[predictedChangeNum] = downCorrectPredictions[predictedChangeNum] / predictionsNum[predictedChangeNum];
            //                relevantHistoriesNum[predictedChangeNum] = 2;
            //            }
            //        }
            //    }

            //    changesRow++;
            //}

            for (int i = 0; i < Constants.DepthsNum * Constants.AnalyzeChangesList.Count; i++)
            {
                if (localAnalyzePosition >= 27888)
                {
                    Console.WriteLine("{0}", localAnalyzePosition);
                }
                else {
                    analyzeResults[localAnalyzePosition] = upPredictionsResults[i];
                    localAnalyzePosition++;
                }
            }

            for (int i = 0; i < Constants.DepthsNum * Constants.AnalyzeChangesList.Count; i++)
            {
                if (localAnalyzePosition >= 27888)
                {
                    Console.WriteLine("{0}", localAnalyzePosition);
                }
                else {
                    analyzeResults[localAnalyzePosition] = downPredictionsResults[i];
                    localAnalyzePosition++;
                }
            }

            return localPredictionRow;
        }

        private double GetPredictionResult(deviceptr<double> predictions, int predictionsRow, AnalyzesDataSet.AnalyzeCombination analyzeCombination)
        {
            double combinedPrediction = 1.0;
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.OpenChange) == AnalyzesDataSet.AnalyzeCombination.OpenChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighChange) == AnalyzesDataSet.AnalyzeCombination.HighChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.LowChange) == AnalyzesDataSet.AnalyzeCombination.LowChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.CloseChange) == AnalyzesDataSet.AnalyzeCombination.CloseChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.VolumeChange) == AnalyzesDataSet.AnalyzeCombination.VolumeChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.VolumeChange] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighLowDif) == AnalyzesDataSet.AnalyzeCombination.HighLowDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighLowDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighOpenDif) == AnalyzesDataSet.AnalyzeCombination.HighOpenDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighOpenDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.LowOpenDif) == AnalyzesDataSet.AnalyzeCombination.LowOpenDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowOpenDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.CloseOpenDif) == AnalyzesDataSet.AnalyzeCombination.CloseOpenDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseOpenDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.HighPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.HighPrevCloseDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighPrevCloseDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.LowPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.LowPrevCloseDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowPrevCloseDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.OpenPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.OpenPrevCloseDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenPrevCloseDif] > Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeOpenChange) == AnalyzesDataSet.AnalyzeCombination.NegativeOpenChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighChange) == AnalyzesDataSet.AnalyzeCombination.NegativeHighChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeLowChange) == AnalyzesDataSet.AnalyzeCombination.NegativeLowChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeCloseChange) == AnalyzesDataSet.AnalyzeCombination.NegativeCloseChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeVolumeChange) == AnalyzesDataSet.AnalyzeCombination.NegativeVolumeChange)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.VolumeChange] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighLowDif) == AnalyzesDataSet.AnalyzeCombination.NegativeHighLowDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighLowDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighOpenDif) == AnalyzesDataSet.AnalyzeCombination.NegativeHighOpenDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighOpenDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeLowOpenDif) == AnalyzesDataSet.AnalyzeCombination.NegativeLowOpenDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowOpenDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeCloseOpenDif) == AnalyzesDataSet.AnalyzeCombination.NegativeCloseOpenDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.CloseOpenDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeHighPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.NegativeHighPrevCloseDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.HighPrevCloseDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeLowPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.NegativeLowPrevCloseDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.LowPrevCloseDif] > -Constants.PredictionErrorRange) ? 1 : 0; }
            if ((analyzeCombination & AnalyzesDataSet.AnalyzeCombination.NegativeOpenPrevCloseDif) == AnalyzesDataSet.AnalyzeCombination.NegativeOpenPrevCloseDif)
            { combinedPrediction *= (predictions[predictionsRow * (int)PredictionsDataSet.DataColumns.NumOfColumns + (int)PredictionsDataSet.DataColumns.OpenPrevCloseDif] > -Constants.PredictionErrorRange) ? 1 : 0; }

            return combinedPrediction;
        }

        #endregion
    }

    internal class GPUAnalyzer : ILGPUModule
    {
        #region Members

        private readonly int m_CombinationsNum = DataAnalyzer.GPUCycleSize;

        private readonly int m_DataSetWidth = (int)DataSet.DataColumns.NumOfColumns;

        private readonly int m_DataSetNumOfRows;

        private readonly int m_PredictionsNum = DataAnalyzer.PredictionItems.Count;

        private readonly int m_AnalyzesNum = DataAnalyzer.AnalyzeItems.Count;

        private readonly double m_ErrorRange = DataAnalyzer.PredictionErrorRange;

        private DeviceMemory<int> m_DataItemsMapDM;
        private DeviceMemory<double> m_DataSetDM;
        private DeviceMemory<int> m_PredictionsDataItemsDM;
        private DeviceMemory<int> m_PredictionsRangesDM;
        private DeviceMemory<int> m_AnalyzesDataItemsDM;
        private DeviceMemory<int> m_AnalyzesRangesDM;
        private DeviceMemory<byte> m_PredictedChangesDM;
        private DeviceMemory<byte> m_ActualChangesDM;
        private DeviceMemory<double> m_AnalyzeResultsDM;

        #endregion

        #region Constructors

        public GPUAnalyzer(double[] dataset, int[] predictionsDataItems, int[] predictionsRanges, int[] analyzesDataItems, int[] analyzesRanges)
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
            m_AnalyzeResultsDM = GPUWorker.Malloc<double>(m_CombinationsNum * m_AnalyzesNum);
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

        public double[] AnalyzeCombinations(int[] combinations, int combinationSize, int combinationsNum, int minimumPredictionsForAnalyze)
        {            
            using (var dCombinations = GPUWorker.Malloc(combinations))
            {
                int numOfThreadsInBlock = 1024;
                int blockY = m_AnalyzesNum;
                int blockX = numOfThreadsInBlock / blockY;
                int gridX = m_CombinationsNum / blockX + 1;
                var block = new dim3(blockX, blockY);
                var grid = new dim3(gridX);
                var lp = new LaunchParam(grid, block);
                GPULaunch(AnalyzeCombinations, lp, dCombinations.Ptr, combinationSize, m_AnalyzeResultsDM.Ptr, 
                    m_PredictedChangesDM.Ptr, m_ActualChangesDM.Ptr, m_AnalyzesRangesDM.Ptr, combinationsNum, minimumPredictionsForAnalyze);
                return m_AnalyzeResultsDM.Gather();
            }
        }

        #endregion

        #region Private Methods

        [Kernel]
        public void BuildPredictedChanges(deviceptr<double> dataSet, deviceptr<byte> predictedChanges, deviceptr<int> dataItemsMap, deviceptr<int> predictionsDataItems, deviceptr<int> predictionsRanges)
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
                deviceptr<double> currentDataSet = dataSet.Ptr((dataRow + range) * m_DataSetWidth);

                currentPredictedChanges[predictionNum] = (dataRow >= m_DataSetNumOfRows - range * 3) ?
                        (byte)0 : currentPredictedChanges[predictionNum] = IsPrediction(currentDataSet, range, columnFrom, columnOf, isDifFromPrevDate, isBigger, -m_ErrorRange, m_ErrorRange);

            }
        }

        [Kernel]
        public void BuildActualChanges(deviceptr<double> dataSet, deviceptr<byte> actualChanges, deviceptr<int> dataItemsMap, deviceptr<int> analyzesDataItems, deviceptr<int> analyzesRanges)
        {
            var dataRow = blockIdx.x * blockDim.x + threadIdx.x;
            deviceptr<byte> currentActualChanges = actualChanges.Ptr(dataRow * m_AnalyzesNum);
            deviceptr<double> currentDataSet = dataSet.Ptr(dataRow * m_DataSetWidth);

            for (int analyzeNum = 0; analyzeNum < m_AnalyzesNum; analyzeNum++)
            {
                int pn = analyzeNum;
                int columnFrom = dataItemsMap[analyzesDataItems[pn] * 4 + 0];
                int columnOf = dataItemsMap[analyzesDataItems[pn] * 4 + 1];
                int isDifFromPrevDate = dataItemsMap[analyzesDataItems[analyzeNum] * 4 + 2];
                int isBigger = dataItemsMap[analyzesDataItems[analyzeNum] * 4 + 3];
                int range = analyzesRanges[analyzeNum];

                currentActualChanges[analyzeNum] = (dataRow >= m_DataSetNumOfRows - range * 3) ?
                    (byte)0 : IsPrediction(currentDataSet, range, columnFrom, columnOf, isDifFromPrevDate, isBigger, m_ErrorRange, -m_ErrorRange);
            }
        }

        [Kernel]
        public void AnalyzeCombinations(deviceptr<int> combinationItems, int combinationSize, deviceptr<double> analyzeResults, 
            deviceptr<byte> predictedChanges, deviceptr<byte> actualChanges, deviceptr<int> analyzeRanges, int combinationsNum, int minimumPredictionsForAnalyze)
        {
            var combinationNum = blockIdx.x * blockDim.x + threadIdx.x;
            var analyzeNum = threadIdx.y;

            if (combinationNum < combinationsNum)
            {
                deviceptr<int> threadCombinationItems = combinationItems.Ptr(combinationNum * combinationSize);

                double predictedChangesSum = 0;
                double actualChangesSum = 0;
                for (int rowNum = 0; rowNum < m_DataSetNumOfRows - analyzeRanges[analyzeNum] * 3; rowNum++)
                {
                    int predictedChange = 1;
                    for (int itemNum = 0; itemNum < combinationSize; itemNum++)
                    {
                        predictedChange *= predictedChanges[rowNum * m_PredictionsNum + threadCombinationItems[itemNum]];
                    }

                    predictedChangesSum += predictedChange;
                    actualChangesSum += predictedChange * actualChanges[rowNum * m_AnalyzesNum + analyzeNum];
                }
                double analyzeResult = (predictedChangesSum > minimumPredictionsForAnalyze) ? actualChangesSum / predictedChangesSum : 0.0;
                analyzeResults[combinationNum * m_AnalyzesNum + analyzeNum] = analyzeResult;
            }
        }

        private byte IsPrediction(deviceptr<double> dataSet, int range, int dataColumFrom, int dataColumOf, int isDifFromPrevDate,  int isBigger, double biggerErrorBorder, double smallerErrorBorder)
        {
            int ofRow = isDifFromPrevDate * range;
            double sumOf = dataSet[(ofRow) * m_DataSetWidth + dataColumOf];
            double sumFrom = dataSet[dataColumFrom];
            //for (int i = 0; i < range; i++)
            //{
            //    sumOf += dataSet[(ofRow) * m_DataSetWidth + dataColumOf];
            //    sumFrom += dataSet[dataColumFrom];
            //}

            return (byte)((isBigger == 1) ?
                (((sumFrom - sumOf) / sumOf / range) > biggerErrorBorder) ? 1 : 0
                :
                (((sumFrom - sumOf) / sumOf / range) < smallerErrorBorder) ? 1 : 0);
        }

        private int[] GetDataItemsMap()
        {
            int mapLength = 4;
            int[] dataItemsMap = new int[DataAnalyzer.DataItems.Count * mapLength];

            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.OpenChange) * mapLength + 0] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.OpenChange) * mapLength + 1] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.OpenChange) * mapLength + 2] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.OpenChange) * mapLength + 3] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.CloseChange) * mapLength + 0] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.CloseChange) * mapLength + 1] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.CloseChange) * mapLength + 2] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.CloseChange) * mapLength + 3] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.VolumeChange) * mapLength + 0] = (int)DataSet.DataColumns.Volume;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.VolumeChange) * mapLength + 1] = (int)DataSet.DataColumns.Volume;            
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.VolumeChange) * mapLength + 2] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.VolumeChange) * mapLength + 3] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.CloseOpenDif) * mapLength + 0] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.CloseOpenDif) * mapLength + 1] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.CloseOpenDif) * mapLength + 2] = 0;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.CloseOpenDif) * mapLength + 3] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.OpenPrevCloseDif) * mapLength + 0] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.OpenPrevCloseDif) * mapLength + 1] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.OpenPrevCloseDif) * mapLength + 2] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.OpenPrevCloseDif) * mapLength + 3] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeOpenChange) * mapLength + 0] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeOpenChange) * mapLength + 1] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeOpenChange) * mapLength + 2] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeOpenChange) * mapLength + 3] = 0;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeCloseChange) * mapLength + 0] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeCloseChange) * mapLength + 1] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeCloseChange) * mapLength + 2] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeCloseChange) * mapLength + 3] = 0;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeVolumeChange) * mapLength + 0] = (int)DataSet.DataColumns.Volume;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeVolumeChange) * mapLength + 1] = (int)DataSet.DataColumns.Volume;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeVolumeChange) * mapLength + 2] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeVolumeChange) * mapLength + 3] = 0;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeCloseOpenDif) * mapLength + 0] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeCloseOpenDif) * mapLength + 1] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeCloseOpenDif) * mapLength + 2] = 0;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeCloseOpenDif) * mapLength + 3] = 0;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeOpenPrevCloseDif) * mapLength + 0] = (int)DataSet.DataColumns.Open;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeOpenPrevCloseDif) * mapLength + 1] = (int)DataSet.DataColumns.Close;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeOpenPrevCloseDif) * mapLength + 2] = 1;
            dataItemsMap[DataAnalyzer.DataItems.IndexOf(DataItem.NegativeOpenPrevCloseDif) * mapLength + 3] = 0;

            return dataItemsMap;
        }

        #endregion
    }
}
