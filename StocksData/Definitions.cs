using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public interface IGPUMethodsMessage
    {
        int MessageSize { get; set; }
        GPUCommand GPUCommand { get; set; }
        GPUMessagesType GPUMessagesType { get; set; }
        int GetSize();

        byte[] GetRowData();
    }
    public enum AckType
    {
        Received,
        NotReceived,
    };

    public enum GPUMessagesType
    {
        AckMessage,
        AnalyzerMethods,
    };

    public enum GPUCommand
    {
        None,
        LoadData,
        ActivateMethod,
        FreeMemory,
        CloseProcess,
    };

    public struct AckMessage : IGPUMethodsMessage
    {
        public int MessageSize { get; set; }
        public GPUCommand GPUCommand { get; set; }

        public GPUMessagesType GPUMessagesType { get; set; }

        public AckType Ack;

        public AckMessage(AckType ack)
        {
            Ack = ack;
            GPUMessagesType = GPUMessagesType.AckMessage;
            GPUCommand = GPUCommand.None;
            MessageSize = 0;
        }

        public byte[] GetRowData()
        {
            IEnumerable<byte> rowData = BitConverter.GetBytes(MessageSize).ToList();
            rowData = rowData.Concat(BitConverter.GetBytes((int)GPUCommand));
            rowData = rowData.Concat(BitConverter.GetBytes((int)GPUMessagesType));
            rowData = rowData.Concat(BitConverter.GetBytes((int)Ack));

            return rowData.ToArray();
        }

        public int GetSize()
        {
            return sizeof(AckType) + sizeof(GPUCommand) + sizeof(GPUMessagesType) + sizeof(int);
        }
    }

    public struct GPUMethodsMessage : IGPUMethodsMessage
    {
        public int MessageSize { get; set; }
        public GPUCommand GPUCommand { get; set; }
        public GPUMessagesType GPUMessagesType { get; set; }
        public int GetSize()
        {
           return sizeof(GPUCommand) + sizeof(GPUMessagesType) + sizeof(int);
        }

        public byte[] GetRowData()
        {
            IEnumerable<byte> rowData = BitConverter.GetBytes(MessageSize).ToList();
            rowData = rowData.Concat(BitConverter.GetBytes((int)GPUCommand));
            rowData = rowData.Concat(BitConverter.GetBytes((int)GPUMessagesType));

            return rowData.ToArray();
        }
    }

    public struct AnalyzerLoadData : IGPUMethodsMessage
    {
        public int MessageSize { get; set; }
        public GPUCommand GPUCommand { get; set; }
        public GPUMessagesType GPUMessagesType { get; set; }
        public int numOfCombinations;
        public int numOfAnalyzeCombinationsItems;
        public int dataSetWidth;
        public int numOfDataSetRows;
        public int predictedCollectionsMaxSize;
        public int numOfPredictedCombinations;
        public int numOfAnalyzesRanges;
        public double predictionErrorRange;
        public double[] dataset;
        public int[] predictedCollections;
        public int[] predictedCollectionsSizes;
        public int[] analyzeCombinationsDataItems;
        public int[] analyzeCombinationsRanges;
        public int[] analyzesRanges;

        public int GetSize()
        {
            int size = sizeof(GPUCommand) + sizeof(GPUMessagesType) + sizeof(int) * 8 + sizeof(double);
            size += dataset.Length * sizeof(double);
            size += predictedCollections.Length * sizeof(int);
            size += predictedCollectionsSizes.Length * sizeof(int);
            size += analyzeCombinationsDataItems.Length * sizeof(int);
            size += analyzeCombinationsRanges.Length * sizeof(int);
            size += analyzesRanges.Length * sizeof(int);
            return size;
        }

        public byte[] GetRowData()
        {
            IEnumerable<byte> rowData = BitConverter.GetBytes(MessageSize).ToList();
            rowData = rowData.Concat(BitConverter.GetBytes((int)GPUCommand));
            rowData = rowData.Concat(BitConverter.GetBytes((int)GPUMessagesType));
            rowData = rowData.Concat(BitConverter.GetBytes(numOfCombinations));
            rowData = rowData.Concat(BitConverter.GetBytes(numOfAnalyzeCombinationsItems));
            rowData = rowData.Concat(BitConverter.GetBytes(dataSetWidth));
            rowData = rowData.Concat(BitConverter.GetBytes(numOfDataSetRows));
            rowData = rowData.Concat(BitConverter.GetBytes(predictedCollectionsMaxSize));
            rowData = rowData.Concat(BitConverter.GetBytes(numOfPredictedCombinations));
            rowData = rowData.Concat(BitConverter.GetBytes(numOfAnalyzesRanges));
            rowData = rowData.Concat(BitConverter.GetBytes(predictionErrorRange));

            byte[] buffer = new byte[GetSize()];
            Buffer.BlockCopy(rowData.ToArray(), 0, buffer, 0, rowData.Count());
            int offset = rowData.Count();
            Buffer.BlockCopy(dataset, 0, buffer, offset, dataset.Length * sizeof(double));
            offset += dataset.Length * sizeof(double);
            Buffer.BlockCopy(predictedCollections, 0, buffer, offset, predictedCollections.Length * sizeof(int));
            offset += predictedCollections.Length * sizeof(int);
            Buffer.BlockCopy(predictedCollectionsSizes, 0, buffer, offset, predictedCollectionsSizes.Length * sizeof(int));
            offset += predictedCollectionsSizes.Length * sizeof(int);
            Buffer.BlockCopy(analyzeCombinationsDataItems, 0, buffer, offset, analyzeCombinationsDataItems.Length * sizeof(int));
            offset += analyzeCombinationsDataItems.Length * sizeof(int);
            Buffer.BlockCopy(analyzeCombinationsRanges, 0, buffer, offset, analyzeCombinationsRanges.Length * sizeof(int));
            offset += analyzeCombinationsRanges.Length * sizeof(int);
            Buffer.BlockCopy(analyzesRanges, 0, buffer, offset, analyzesRanges.Length * sizeof(int));
            offset += analyzesRanges.Length * sizeof(int);

            return buffer;
        }
    };

    public struct AnalyzerActivateData : IGPUMethodsMessage
    {
        public int MessageSize { get; set; }
        public GPUCommand GPUCommand { get; set; }
        public GPUMessagesType GPUMessagesType { get; set; }
        public ulong[] Combinations;

        public int GetSize()
        {
            int size = sizeof(GPUCommand) + sizeof(GPUMessagesType) + sizeof(int);
            size += Combinations.Length * sizeof(ulong);
            return size;
        }

        public byte[] GetRowData()
        {
            IEnumerable<byte> rowData = BitConverter.GetBytes(MessageSize).ToList();
            rowData = rowData.Concat(BitConverter.GetBytes((int)GPUCommand));
            rowData = rowData.Concat(BitConverter.GetBytes((int)GPUMessagesType));

            byte[] buffer = new byte[GetSize()];
            Buffer.BlockCopy(rowData.ToArray(), 0, buffer, 0, rowData.Count());
            int offset = rowData.Count();
            Buffer.BlockCopy(Combinations, 0, buffer, offset, Combinations.Length * sizeof(ulong));

            return buffer;
        }
    };
}
