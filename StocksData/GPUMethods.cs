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
            int changesRow = 0;

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

    internal class GPUAnalyzer// : ILGPUModule
    {
        //[DllImport("C:\\Ekans\\Stocks\\StocksGenius\\Debug\\GPUMethods.exe", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        //public static extern void LoadAnalyzer([In] double[] dataset,
        //                                             [In] int[] predictedCollections,
        //                                             [In] int[] predictedCollectionsSizes,
        //                                             [In] int[] analyzeCombinationsDataItems,
        //                                             [In] int[] analyzeCombinationsRanges,
        //                                             [In] int[] analyzesRanges,
        //                                             int numOfCombinations,
        //                                             int numOfAnalyzeCombinationsItems,
        //                                             int dataSetWidth,
        //                                             int numOfDataSetRows,
        //                                             int predictedCollectionsMaxSize,
        //                                             int numOfPredictedCombinations,
        //                                             int numOfAnalyzesRanges,
        //                                             double predictionErrorRange);

        //[DllImport("C:\\Ekans\\Stocks\\StocksGenius\\Debug\\GPUMethods.exe", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        //public static extern void AnalyzeCombinations([In] ulong[] dataset, [Out] double[] analyzeResults);

        //[DllImport("C:\\Ekans\\Stocks\\StocksGenius\\Debug\\GPUMethods.exe", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        //public static extern void UnloadAnalyzer();


        #region Members

        private static TcpClient tcpClient;

        private static NetworkStream netStream;

        private readonly int m_CombinationsNum = DataAnalyzer.GPUCycleSize;

        private readonly int m_DataSetWidth = (int)DataSet.DataColumns.NumOfColumns;

        private readonly int m_DataSetNumOfRows;

        private readonly int m_AnalyzesNum = DataAnalyzer.AnalyzeCombinationItems.Count;

        private readonly int m_PredictedCollectionsMaxSize;

        private readonly int m_NumOfCombinationItems = DataAnalyzer.CombinationItems.Count;

        private readonly int[] m_AnalyzeRanges = DataAnalyzer.AnalyzeRanges.ToArray();

        private readonly int m_AnalyzeMaxCombinationSize = DataAnalyzer.AnalyzeMaxCombinationSize;

        private const int NumOfThreads = 32;
        
        private readonly DeviceMemory<double> m_DataSet;
        private readonly DeviceMemory<int> m_PredictedCollections;
        private readonly DeviceMemory<int> m_PredictedCollectionsSizes;
        private readonly DeviceMemory<int> m_AnalyzeCombinationsDataItems;
        private readonly DeviceMemory<int> m_AnalyzeCombinationsRanges;
        private readonly DeviceMemory<double> m_AnalyzeResults;
        private readonly DeviceMemory<int> m_CorrectPredictions;
        private readonly DeviceMemory<int> m_CombinationItems;
        private readonly DeviceMemory<int> m_PredictionPtrs;


        #endregion

        #region Constructors
        static GPUAnalyzer()
        {
            tcpClient = new TcpClient(new IPEndPoint(IPAddress.Loopback, 10012));
            tcpClient.SendBufferSize = 1 << 27;
            tcpClient.ReceiveBufferSize = 1 << 27;
            tcpClient.Connect(new IPEndPoint(IPAddress.Loopback, 10011));
            netStream = tcpClient.GetStream();
            netStream.ReadTimeout = 100;
        }

        //public GPUAnalyzer(double[] dataset, int[] predictedCollections, int[] predictedCollectionsSizes, int[] analyzeCombinationsDataItems, int[] analyzeCombinationsRanges)
        //    : base(GPUModuleTarget.DefaultWorker)
        //{
        //    m_DataSetNumOfRows = dataset.Length / m_DataSetWidth;
        //    m_PredictedCollectionsMaxSize = predictedCollections.Length / predictedCollectionsSizes.Length;
        //    m_DataSet = GPUWorker.Malloc(dataset);
        //    m_PredictedCollections = GPUWorker.Malloc(predictedCollections);
        //    m_PredictedCollectionsSizes = GPUWorker.Malloc(predictedCollectionsSizes);
        //    m_AnalyzeCombinationsDataItems = GPUWorker.Malloc<int>(analyzeCombinationsDataItems);
        //    m_AnalyzeCombinationsRanges = GPUWorker.Malloc<int>(analyzeCombinationsRanges);
        //    m_AnalyzeResults = GPUWorker.Malloc<double>(m_AnalyzesNum * m_CombinationsNum);
        //    m_CorrectPredictions = GPUWorker.Malloc<int>(NumOfThreads * m_AnalyzesNum);
        //    m_CombinationItems = GPUWorker.Malloc<int>(NumOfThreads * m_AnalyzeMaxCombinationSize);
        //    m_PredictionPtrs = GPUWorker.Malloc<int>(NumOfThreads * m_AnalyzeMaxCombinationSize);
        //}

        public GPUAnalyzer(double[] dataset,
                           int[] predictedCollections,
                           int[] predictedCollectionsSizes,
                           int[] analyzeCombinationsDataItems,
                           int[] analyzeCombinationsRanges,
                           int[] analyzesRanges,
                           int numOfCombinations,
                           int numOfAnalyzeCombinationsItems,
                           int dataSetWidth,
                           int numOfDataSetRows,
                           int predictedCollectionsMaxSize,
                           int numOfPredictedCombinations,
                           int numOfAnalyzesRanges,
                           double predictionErrorRange)
        //   : base(GPUModuleTarget.DefaultWorker)
        {
            AnalyzerLoadData loadData = new AnalyzerLoadData();
            loadData.GPUCommand = GPUCommand.LoadData;
            loadData.GPUMessagesType = GPUMessagesType.AnalyzerMethods;
            loadData.dataset = dataset;
            loadData.predictedCollections = predictedCollections;
            loadData.predictedCollectionsSizes = predictedCollectionsSizes;
            loadData.analyzeCombinationsDataItems = analyzeCombinationsDataItems;
            loadData.analyzeCombinationsRanges = analyzeCombinationsRanges;
            loadData.analyzesRanges = analyzesRanges;
            loadData.numOfCombinations = numOfCombinations;
            loadData.numOfAnalyzeCombinationsItems = numOfAnalyzeCombinationsItems;
            loadData.dataSetWidth = dataSetWidth;
            loadData.numOfDataSetRows = numOfDataSetRows;
            loadData.predictedCollectionsMaxSize = predictedCollectionsMaxSize;
            loadData.numOfPredictedCombinations = numOfPredictedCombinations;
            loadData.numOfAnalyzesRanges = numOfAnalyzesRanges;
            loadData.predictionErrorRange = predictionErrorRange;

            SendMessage(loadData);
        }

        #endregion

        #region Interface

        public void AnalyzeCombinations(ulong[] combinations, ref double[] analyzeResults)
        {
            AnalyzerActivateData activateData = new AnalyzerActivateData();
            activateData.GPUCommand = GPUCommand.ActivateMethod;
            activateData.GPUMessagesType = GPUMessagesType.AnalyzerMethods;
            activateData.Combinations = combinations;
             
            SendMessage(activateData);
            ReceiveMessage(ref analyzeResults);
        }

        public void Unload()
        {
            GPUMethodsMessage message = new GPUMethodsMessage();
            message.GPUCommand = GPUCommand.FreeMemory;
            message.GPUMessagesType = GPUMessagesType.AnalyzerMethods;

            SendMessage(message);
        }

        private void SendMessage(IGPUMethodsMessage message)
        {
            int size = message.GetSize();
            message.MessageSize = size;
            int rowSize = 50000;
            int numOfRows = size / rowSize + 1;
            byte[] buffer = message.GetRowData();
            AckMessage ack = new AckMessage(AckType.NotReceived);
            do
            {
                netStream.Write(buffer, 0, size); 

                ack = WaitForAck();
            } while (ack.Ack != AckType.Received);
        }

        private AckMessage WaitForAck()
        {
            AckMessage ack = new AckMessage(AckType.NotReceived);
            do
            {
                byte[] ackBuffer = new byte[ack.GetSize()];

                try
                {
                    int received = netStream.Read(ackBuffer, 0, ackBuffer.Length);
                    if (received != ack.GetSize())
                    {
                        return ack;
                    }

                    ack.MessageSize = BitConverter.ToInt32(ackBuffer, 0);
                    ack.GPUCommand = (GPUCommand)BitConverter.ToInt32(ackBuffer, 4);
                    ack.GPUMessagesType = (GPUMessagesType)BitConverter.ToInt32(ackBuffer, 8);
                    ack.Ack = (AckType)BitConverter.ToInt32(ackBuffer, 12);
                    if (ack.Ack == AckType.Received)
                    {
                        //Console.WriteLine("Ack Received");
                        return ack;
                    }
                }
                catch (Exception e)
                {

                    return ack;
                }

            } while (true);

            return ack;
        }
        void SendAck(AckType ack)
        {
            AckMessage ackMessage = new AckMessage(ack);
            netStream.Write(ackMessage.GetRowData(), 0, ackMessage.GetSize());
        }

        private void ReceiveMessage(ref double[] analyzeResults)
        {
            byte[] resultBuffer = new byte[1 << 27];
            while (true)
            {
                try
                {
                    int received = netStream.Read(resultBuffer, 0, 1 << 27);
                    int size = BitConverter.ToInt32(resultBuffer, 0);
                    if (received != size)
                    {
                        SendAck(AckType.NotReceived);
                    }
                    else
                    {
                        SendAck(AckType.Received);
                        break;
                    }
                }
                catch (Exception e)
                {                    
                }
            }

            Buffer.BlockCopy(resultBuffer, 4, analyzeResults, 0, analyzeResults.Length);

            //for (int i = 0; i < analyzeResults.Length; i++)
            //{
            //    analyzeResults[i] = BitConverter.ToDouble(resultBuffer, i * 8 + 4);
            //}

            //while (!receiveClient.Client.Connected)
            //{
            //    //wait
            //}
            //IPEndPoint rec = new IPEndPoint(IPAddress.Any, 0);

            //byte[] resultBuffer = receiveClient.Receive(ref rec);
            //int size = BitConverter.ToInt32(resultBuffer, 0);

            //int received = resultBuffer.Length;
            //while (received < size)
            //{
            //    resultBuffer = resultBuffer.Concat(receiveClient.Receive(ref rec)).ToArray();
            //}

            //for (int i = 0; i < analyzeResults.Length; i++)
            //{
            //    analyzeResults[i] = BitConverter.ToDouble(resultBuffer, i * 8 + 4);
            //}
        }

        private void UnloadAnalyzer()
        {
            throw new NotImplementedException();
        }

        public static double[] AnalyzeCombinations(ulong[] combinations, double[] dataset, int[] predictedCollections, int[] predictedCollectionsSizes, int[] analyzeCombinationsDataItems, int[] analyzeCombinationsRanges)
        {
            //GPUMethodsAnalyzer(combinations, dataset, predictedCollections, predictedCollectionsSizes, analyzeCombinationsDataItems, analyzeCombinationsRanges);
            //using (var loadedModule = new GPUAnalyzer(dataset, predictedCollections, predictedCollectionsSizes, analyzeCombinationsDataItems, analyzeCombinationsRanges))
            //{
            //      return loadedModule.AnalyzerMap(combinations, dataset, predictedCollections, predictedCollectionsSizes, analyzeCombinationsDataItems, analyzeCombinationsRanges);
            //}
            return new double[0];
        }

        //public double[] AnalyzeCombinations(ulong[] combinations, double[] dataset, int[] predictedCollections, int[] predictedCollectionsSizes, int[] analyzeCombinationsDataItems, int[] analyzeCombinationsRanges)
        //{
        //    return AnalyzerMap(combinations, dataset, predictedCollections, predictedCollectionsSizes, analyzeCombinationsDataItems, analyzeCombinationsRanges);
        //}

        //public double[] AnalyzerMap(ulong[] combinations, double[] dataset, int[] predictedCollections, int[] predictedCollectionsSizes, int[] analyzeCombinationsDataItems, int[] analyzeCombinationsRanges)
        //{
        //    using (var dDataSet = GPUWorker.Malloc(dataset))
        //    using (var dPredictedCollections = GPUWorker.Malloc(predictedCollections))
        //    using (var dPredictedCollectionsSizes = GPUWorker.Malloc(predictedCollectionsSizes))
        //    using (var dAnalyzeCombinationsDataItems = GPUWorker.Malloc<int>(analyzeCombinationsDataItems))
        //    using (var dAnalyzeCombinationsRanges = GPUWorker.Malloc<int>(analyzeCombinationsRanges))
        //    using (var dAnalyzeResults = GPUWorker.Malloc<double>(m_AnalyzesNum * m_CombinationsNum))
        //    using (var dCorrectPredictions = GPUWorker.Malloc<int>(NumOfThreads * m_AnalyzesNum))
        //    using (var dCombinationItems = GPUWorker.Malloc<int>(NumOfThreads * m_AnalyzeMaxCombinationSize))
        //    using (var dPredictionPtrs = GPUWorker.Malloc<int>(NumOfThreads * m_AnalyzeMaxCombinationSize))
        //    using (var dCombinations = GPUWorker.Malloc(combinations))
        //    {
        //        var block = new dim3(NumOfThreads);
        //        var grid = new dim3(combinations.Length / block.x);
        //        var lp = new LaunchParam(grid, block);
        //        GPULaunch(AnalyzerKernel, lp, dCombinations.Ptr, dDataSet.Ptr, dPredictedCollections.Ptr, dPredictedCollectionsSizes.Ptr, dAnalyzeCombinationsDataItems.Ptr,
        //            dAnalyzeCombinationsRanges.Ptr, dAnalyzeResults.Ptr, dCorrectPredictions.Ptr, dCombinationItems.Ptr, dPredictionPtrs.Ptr);
        //        return m_AnalyzeResults.Gather();
        //    }
        //}

        //public double[] AnalyzerMap(ulong[] combinations)
        //{

        //    using (var dCombinations = GPUWorker.Malloc(combinations))
        //    {
        //        var block = new dim3(NumOfThreads);
        //        var grid = new dim3(combinations.Length / block.x);
        //        var lp = new LaunchParam(grid, block);
        //        GPULaunch(AnalyzerKernel, lp, dCombinations.Ptr, m_DataSet.Ptr, m_PredictedCollections.Ptr, m_PredictedCollectionsSizes.Ptr, m_AnalyzeCombinationsDataItems.Ptr,
        //            m_AnalyzeCombinationsRanges.Ptr, m_AnalyzeResults.Ptr, m_CorrectPredictions.Ptr, m_CombinationItems.Ptr, m_PredictionPtrs.Ptr);
        //        return m_AnalyzeResults.Gather();
        //    }
        //}

        //[Kernel]
        //public void AnalyzerKernel(deviceptr<ulong> combinations, deviceptr<double> dataSet, deviceptr<int> predictedCollections, deviceptr<int> predictedCollectionsSizes, deviceptr<int> analyzeCombinationsDataItems,
        //            deviceptr<int> analyzeCombinationsRanges, deviceptr<double> analyzeResults, deviceptr<int> correctPredictions, deviceptr<int> combinationItems, deviceptr<int> predictionPtrs)
        //{
        //    var combinationStart = blockIdx.x * blockDim.x + threadIdx.x;
        //    var combinationStride = gridDim.x * blockDim.x;
        //    int currentThread = threadIdx.x;

        //    for (var combinationNum = combinationStart; combinationNum < m_CombinationsNum; combinationNum += combinationStride)
        //    {
        //        ulong combination = combinations[combinationNum];
        //        int resultsStart = combinationNum * m_AnalyzesNum;

        //        int combinationSize = 0;
        //        int combinationItem = 0;
        //        for (int i = 0; i < m_NumOfCombinationItems; i++)
        //        {
        //            if ((combination & ((ulong)1 << i)) != 0)
        //            {
        //                combinationSize++;
        //                combinationItem = i;
        //            }
        //        }

        //        if (combinationSize == 1)
        //        {
        //            int combinationRowPtr = 0;
        //            int numOfCombinations = 0;
        //            int combinationRow = 0;

        //            while (combinationRowPtr < predictedCollectionsSizes[combinationItem])
        //            {
        //                combinationRow = predictedCollections[combinationItem * m_PredictedCollectionsMaxSize + combinationRowPtr];

        //                for (int analyzeCombinationNum = 0; analyzeCombinationNum < m_AnalyzesNum; analyzeCombinationNum++)
        //                {
        //                    int dataItem = analyzeCombinationsDataItems[analyzeCombinationNum];
        //                    int range = analyzeCombinationsRanges[analyzeCombinationNum];

        //                    if (combinationRow > m_DataSetNumOfRows - range * 2)
        //                    {
        //                        break;
        //                    }

        //                    bool isDifFromCurrentDate = false;
        //                    DataSet.DataColumns dataColumOf = DataSet.DataColumns.Open;
        //                    DataSet.DataColumns dataColumFrom = DataSet.DataColumns.Open;
        //                    bool bigger = true;
        //                    double errorRange = DataAnalyzer.PredictionErrorRange;

        //                    if ((DataItem)dataItem == DataItem.OpenChange)
        //                    {
        //                        isDifFromCurrentDate = false;
        //                        dataColumOf = DataSet.DataColumns.Open;
        //                        dataColumFrom = DataSet.DataColumns.Open;
        //                    }
        //                    else if ((DataItem)dataItem == DataItem.CloseChange)
        //                    {
        //                        isDifFromCurrentDate = false;
        //                        dataColumOf = DataSet.DataColumns.Close;
        //                        dataColumFrom = DataSet.DataColumns.Close;
        //                    }
        //                    else if ((DataItem)dataItem == DataItem.VolumeChange)
        //                    {
        //                        isDifFromCurrentDate = false;
        //                        dataColumOf = DataSet.DataColumns.Close;
        //                        dataColumFrom = DataSet.DataColumns.Close;
        //                    }
        //                    else if ((DataItem)dataItem == DataItem.CloseOpenDif)
        //                    {
        //                        isDifFromCurrentDate = true;
        //                        dataColumOf = DataSet.DataColumns.Close;
        //                        dataColumFrom = DataSet.DataColumns.Open;
        //                    }
        //                    else if ((DataItem)dataItem == DataItem.OpenPrevCloseDif)
        //                    {
        //                        isDifFromCurrentDate = false;
        //                        dataColumOf = DataSet.DataColumns.Open;
        //                        dataColumFrom = DataSet.DataColumns.Close;
        //                    }
        //                    else if ((DataItem)dataItem == DataItem.NegativeOpenChange)
        //                    {
        //                        isDifFromCurrentDate = false;
        //                        dataColumOf = DataSet.DataColumns.Open;
        //                        dataColumFrom = DataSet.DataColumns.Open;
        //                        bigger = false;
        //                        errorRange = -DataAnalyzer.PredictionErrorRange;
        //                    }
        //                    else if ((DataItem)dataItem == DataItem.NegativeCloseChange)
        //                    {
        //                        isDifFromCurrentDate = false;
        //                        dataColumOf = DataSet.DataColumns.Close;
        //                        dataColumFrom = DataSet.DataColumns.Close;
        //                        bigger = false;
        //                        errorRange = -DataAnalyzer.PredictionErrorRange;
        //                    }
        //                    else if ((DataItem)dataItem == DataItem.NegativeVolumeChange)
        //                    {
        //                        isDifFromCurrentDate = false;
        //                        dataColumOf = DataSet.DataColumns.Close;
        //                        dataColumFrom = DataSet.DataColumns.Close;
        //                        bigger = false;
        //                        errorRange = -DataAnalyzer.PredictionErrorRange;
        //                    }
        //                    else if ((DataItem)dataItem == DataItem.NegativeCloseOpenDif)
        //                    {
        //                        isDifFromCurrentDate = true;
        //                        dataColumOf = DataSet.DataColumns.Close;
        //                        dataColumFrom = DataSet.DataColumns.Open;
        //                        bigger = false;
        //                        errorRange = -DataAnalyzer.PredictionErrorRange;
        //                    }
        //                    else if ((DataItem)dataItem == DataItem.NegativeOpenPrevCloseDif)
        //                    {
        //                        isDifFromCurrentDate = false;
        //                        dataColumOf = DataSet.DataColumns.Open;
        //                        dataColumFrom = DataSet.DataColumns.Close;
        //                        bigger = false;
        //                        errorRange = -DataAnalyzer.PredictionErrorRange;
        //                    }

        //                    int dataOfStartPosition = isDifFromCurrentDate ? 0 : range;
        //                    double sumOf = 0.0;
        //                    double sumFrom = 0.0;
        //                    for (int i = combinationRow; i < combinationRow + range; i++)
        //                    {
        //                        sumOf += dataSet[(combinationRow + dataOfStartPosition) * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumOf];
        //                        sumFrom += dataSet[combinationRow * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumFrom];
        //                    }

        //                    double actualChange = (sumFrom - sumOf) / sumOf / range;

        //                    if ((bigger && actualChange > errorRange) || (!bigger && actualChange < errorRange))
        //                    {
        //                        correctPredictions[analyzeCombinationNum]++;
        //                    }
        //                }

        //                numOfCombinations++;
        //                combinationRowPtr++;
        //            }

        //            for (int i = 0; i < m_AnalyzesNum; i++)
        //            {
        //                if (numOfCombinations > 0)
        //                {
        //                    analyzeResults[resultsStart + i] = correctPredictions[i] / numOfCombinations;
        //                }
        //            }
        //        }
        //        else
        //        {
        //            int currentItem = 0;
        //            for (int i = 0; i < m_NumOfCombinationItems; i++)
        //            {
        //                if ((combination & ((ulong)1 << i)) != 0)
        //                {
        //                    combinationItems[currentThread * m_AnalyzeMaxCombinationSize + currentItem] = i;
        //                    predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize + currentItem] = 0;
        //                    currentItem++;
        //                }
        //            }

        //            //DoAnalyzerCombination(resultsStart, combinationSize, currentThread);
        //        }
        //    }
        //    }

        //private void DoAnalyzerOneSizeCombination(int resultsStart, int combinationItem, int currentThread, deviceptr<double> dataSet, deviceptr<int> predictedCollections, deviceptr<int> predictedCollectionsSizes, deviceptr<int> analyzeCombinationsDataItems,
        //            deviceptr<int> analyzeCombinationsRanges, deviceptr<double> analyzeResults, deviceptr<int> correctPredictions, deviceptr<int> combinationItems, deviceptr<int> predictionPtrs)
        //{
        //    int combinationRowPtr = 0;
        //    int numOfCombinations = 0;
        //    int combinationRow = 0;

        //    while (combinationRowPtr < predictedCollectionsSizes[combinationItem])
        //    {
        //        combinationRow = predictedCollections[combinationItem * m_PredictedCollectionsMaxSize + combinationRowPtr];

        //        for (int analyzeCombinationNum = 0; analyzeCombinationNum < m_AnalyzesNum; analyzeCombinationNum++)
        //        {
        //            int dataItem = analyzeCombinationsDataItems[analyzeCombinationNum];
        //            int range = analyzeCombinationsRanges[analyzeCombinationNum];

        //            if (combinationRow > m_DataSetNumOfRows - range * 2)
        //            {
        //                continue;
        //            }

        //            if (IsContainsPrediction(dataSet, dataItem, range, combinationRow, DataAnalyzer.PredictionErrorRange, -DataAnalyzer.PredictionErrorRange))
        //            {
        //                correctPredictions[currentThread * m_AnalyzesNum + analyzeCombinationNum]++;
        //            }
        //        }

        //        numOfCombinations++;
        //        combinationRowPtr++;
        //    }

        //    for (int i = 0; i < m_AnalyzesNum; i++)
        //    {
        //        if (numOfCombinations > 0)
        //        {
        //            analyzeResults[resultsStart + i] = correctPredictions[currentThread * m_AnalyzesNum + i] / numOfCombinations;
        //        }
        //    }
        //}

        //private void DoAnalyzerCombination(int resultsStart, int combinationSize, int currentThread, deviceptr<double> dataSet, deviceptr<int> predictedCollections, deviceptr<int> predictedCollectionsSizes, deviceptr<int> analyzeCombinationsDataItems,
        //            deviceptr<int> analyzeCombinationsRanges, deviceptr<double> analyzeResults, deviceptr<int> correctPredictions, deviceptr<int> combinationItems, deviceptr<int> predictionPtrs)
        //{
        //    int combinationRow = 0;
        //    int numOfCombinations = 0;

        //    while ((combinationRow = GetNextCombinationRow(combinationSize, currentThread, dataSet, predictedCollections, predictedCollectionsSizes, analyzeCombinationsDataItems,
        //            analyzeCombinationsRanges, analyzeResults, correctPredictions, combinationItems, predictionPtrs)) != -1)
        //    {
        //        for (int analyzeCombinationNum = 0; analyzeCombinationNum < m_AnalyzesNum; analyzeCombinationNum++)
        //        {
        //            int dataItem = analyzeCombinationsDataItems[analyzeCombinationNum];
        //            int range = analyzeCombinationsRanges[analyzeCombinationNum];

        //            if (combinationRow > m_DataSetNumOfRows - range * 2)
        //            {
        //                continue;
        //            }

        //            if (IsContainsPrediction(dataSet, dataItem, range, combinationRow, DataAnalyzer.PredictionErrorRange, -DataAnalyzer.PredictionErrorRange))
        //            {
        //                correctPredictions[currentThread * m_AnalyzesNum + analyzeCombinationNum]++;
        //            }
        //        }
        //        numOfCombinations++;
        //    }

        //    for (int i = 0; i < m_AnalyzesNum; i++)
        //    {
        //        analyzeResults[resultsStart + i] = correctPredictions[currentThread * m_AnalyzesNum + i] / numOfCombinations;
        //    }
        //}

        private int GetRangeNum(ulong analyzeCombinationItem)
        {
            for (int i = 1; i < m_AnalyzeRanges.Length; i++)
            {
                if (((ulong)1 << (10 * i) > analyzeCombinationItem))
                {
                    return i - 1;
                }
            }

            return m_AnalyzeRanges.Length - 1;
        }

        private int GetNextCombinationRow(int combinationSize, int currentThread, deviceptr<double> dataSet, deviceptr<int> predictedCollections, deviceptr<int> predictedCollectionsSizes, deviceptr<int> analyzeCombinationsDataItems,
                    deviceptr<int> analyzeCombinationsRanges, deviceptr<double> analyzeResults, deviceptr<int> correctPredictions, deviceptr<int> combinationItems, deviceptr<int> predictionPtrs)
        {
            while (predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize] < predictedCollectionsSizes[combinationItems[currentThread * m_AnalyzeMaxCombinationSize]])
            {
                for (int i = 0; i < combinationSize - 1; i++)
                {
                    int j = i + 1;
                    while (predictedCollections[combinationItems[currentThread * m_AnalyzeMaxCombinationSize + i] * m_PredictedCollectionsMaxSize + predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize + i]] > predictedCollections[combinationItems[currentThread * m_AnalyzeMaxCombinationSize + j] * m_PredictedCollectionsMaxSize +predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize + j]])
                    {
                        predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize + j]++;

                        if (predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize + j] < predictedCollectionsSizes[combinationItems[currentThread * m_AnalyzeMaxCombinationSize + j]])
                        {
                            return -1;
                        }
                    }
                }

                bool match = true;
                for (int i = 0; i < combinationSize - 1; i++)
                {
                    int j = i + 1;
                    if (predictedCollections[combinationItems[currentThread * m_AnalyzeMaxCombinationSize + i] * m_PredictedCollectionsMaxSize + predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize + i]] != predictedCollections[combinationItems[currentThread * m_AnalyzeMaxCombinationSize + j] * m_PredictedCollectionsMaxSize + predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize + j]])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                {
                    int matchRow =predictedCollections[combinationItems[currentThread * m_AnalyzeMaxCombinationSize] * m_PredictedCollectionsMaxSize + predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize]];
                    predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize]++;

                    return matchRow;
                }

                predictionPtrs[currentThread * m_AnalyzeMaxCombinationSize]++;
            }
            return -1;
        }

        public bool IsContainsPrediction(deviceptr<double> dataSet, int dataItem, int range, int dataRow, double upperErrorBorder, double lowerErrorBorder)
        {
            if ((DataItem)dataItem == DataItem.OpenChange
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Open, DataSet.DataColumns.Open, false) > upperErrorBorder)
            { return true; }
            if ((DataItem)dataItem == DataItem.CloseChange
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Close, DataSet.DataColumns.Close, false) > upperErrorBorder)
            { return true; }
            if ((DataItem)dataItem == DataItem.VolumeChange
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, false) > upperErrorBorder)
            { return true; }
            if ((DataItem)dataItem == DataItem.CloseOpenDif
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Close, DataSet.DataColumns.Open, true) > upperErrorBorder)
            { return true; }
            if ((DataItem)dataItem == DataItem.OpenPrevCloseDif
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Open, DataSet.DataColumns.Close, false) > upperErrorBorder)
            { return true; }
            if ((DataItem)dataItem == DataItem.NegativeOpenChange
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Open, DataSet.DataColumns.Open, false) < lowerErrorBorder)
            { return true; }
            if ((DataItem)dataItem == DataItem.NegativeCloseChange
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Close, DataSet.DataColumns.Close, false) < lowerErrorBorder)
            { return true; }
            if ((DataItem)dataItem == DataItem.NegativeVolumeChange
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Volume, DataSet.DataColumns.Volume, false) < lowerErrorBorder)
            { return true; }
            if ((DataItem)dataItem == DataItem.NegativeCloseOpenDif
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Close, DataSet.DataColumns.Open, true) < lowerErrorBorder)
            { return true; }
            if ((DataItem)dataItem == DataItem.NegativeOpenPrevCloseDif
                && CalculateChange(dataSet, dataRow, range, DataSet.DataColumns.Open, DataSet.DataColumns.Close, false) < lowerErrorBorder)
            { return true; }

            return false;
        }

        private double CalculateChange(deviceptr<double> dataSet, int dataRow, int range, DataSet.DataColumns dataColumFrom, DataSet.DataColumns dataColumOf, bool isDifFromCurrentDate)
        {
            int dataOfStartPosition = isDifFromCurrentDate ? 0 : range;
            double sumOf = 0.0;
            double sumFrom = 0.0;
            for (int i = dataRow; i < dataRow + range; i++)
            {
                sumOf += dataSet[(dataRow + dataOfStartPosition) * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumOf];
                sumFrom += dataSet[dataRow * (int)DataSet.DataColumns.NumOfColumns + (int)dataColumFrom];
            }

            return (sumFrom - sumOf) / sumOf / range;
        }

        #endregion
    }
}
